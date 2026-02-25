from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket

from embedding_cluster.indexer import main_indexer
from embedding_cluster.server.models import (
    IndexRequest,
    IndexStartResponse,
    IndexStatusResponse,
    MessageResponse,
)
from embedding_cluster.server.tasks import TaskState, TaskStatus, task_registry
from embedding_cluster.server.ws import ws_manager
from embedding_cluster.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/index", tags=["index"])


def resolve_csv_path(csv_filename: str) -> Path:
    candidate = Path(csv_filename)
    if ".." in candidate.parts:
        msg = "CSV filename must not contain parent directory references"
        raise ValueError(msg)

    if candidate.is_absolute():
        msg = "Absolute CSV paths are not allowed"
        raise ValueError(msg)

    if candidate.parts[:1] == ("uploads",) or candidate.parts[:2] == (".", "uploads"):
        return candidate

    return Path("./uploads") / candidate


def _get_collection_names(settings: Settings) -> list[str]:
    """Build collection names from settings."""
    names: list[str] = []
    prefix = settings.chromadb_collection_prefix
    if settings.image_embedding_fields:
        for field in settings.image_embedding_fields:
            names.append(f"{prefix}{field}")
    if settings.text_embedding_fields:
        for field in settings.text_embedding_fields:
            names.append(f"{prefix}{field}")
    return names


async def _run_indexing(task_state: TaskState, request: IndexRequest) -> None:
    """Run indexing in background, updating task state and broadcasting progress."""
    heartbeat_task: asyncio.Task[None] | None = None
    try:
        # Construct Settings from IndexRequest
        try:
            csv_filename = resolve_csv_path(request.csv_filename)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        settings = Settings(
            running_mode="INDEX",
            local_csv_filename=str(csv_filename),
            id_field=request.id_field,
            image_embedding_fields=request.image_embedding_fields,
            text_embedding_fields=request.text_embedding_fields,
            image_model_name=request.image_model_name,
            text_model_name=request.text_model_name,
            chromadb_collection_prefix=request.chromadb_collection_prefix,
            number_of_async_tasks=request.number_of_async_tasks,
            index_bulk_size=request.index_bulk_size,
            index_start_line=request.index_start_line,
            index_end_line=request.index_end_line,
            process_unit_device=request.process_unit_device,
            embedding_fields_prefix=request.embedding_fields_prefix,
        )

        # Update task status to RUNNING
        task_state.status = TaskStatus.RUNNING
        start_time = time.monotonic()

        # Define progress callback
        def on_progress(progress_data: dict[str, Any]) -> None:
            if progress_data.get("total_rows") is None:
                progress_data["total_rows"] = total_rows
            progress_data["status"] = task_state.status.value
            progress_data["type"] = "progress"
            task_state.progress = progress_data
            # Fire and forget broadcast (intentionally not awaited)
            # ruff: noqa: RUF006
            asyncio.create_task(ws_manager.broadcast(task_state.job_id, progress_data))

        # Define log callback
        async def on_log(message: str, level: str, verbosity: str) -> None:
            await ws_manager.broadcast(
                task_state.job_id,
                {
                    "type": "log",
                    "level": level,
                    "message": message,
                    "verbosity": verbosity,
                },
            )

        # Heartbeat background task
        async def _heartbeat() -> None:
            while True:
                await asyncio.sleep(3)
                elapsed = time.monotonic() - start_time
                await ws_manager.broadcast(
                    task_state.job_id,
                    {
                        "type": "heartbeat",
                        "elapsed_seconds": elapsed,
                    },
                )

        heartbeat_task = asyncio.create_task(_heartbeat())

        total_rows = request.total_rows
        on_progress(
            {
                "rows_indexed": 0,
                "total_rows": total_rows,
                "errors": 0,
                "elapsed_seconds": 0,
            }
        )

        # Run indexer with callbacks and cancel event
        await main_indexer(
            settings,
            on_progress=on_progress,
            on_log=on_log,
            cancel_event=task_state.cancel_event,
        )

        # Success — send completion message
        task_state.status = TaskStatus.COMPLETED
        elapsed = time.monotonic() - start_time
        collection_names = _get_collection_names(settings)
        rows_indexed = task_state.progress.get("rows_indexed", 0)
        # ruff: noqa: RUF006
        asyncio.create_task(
            ws_manager.broadcast(
                task_state.job_id,
                {
                    "type": "completed",
                    "status": "completed",
                    "progress": task_state.progress,
                    "total_indexed": rows_indexed,
                    "collection_names": collection_names,
                    "elapsed_seconds": elapsed,
                },
            )
        )
    except Exception as e:
        logger.exception("Indexing failed for job %s", task_state.job_id)
        task_state.status = TaskStatus.FAILED
        task_state.error = str(e)
        # ruff: noqa: RUF006
        asyncio.create_task(
            ws_manager.broadcast(
                task_state.job_id,
                {
                    "type": "error",
                    "status": task_state.status.value,
                    "error": task_state.error,
                    "message": str(e),
                    "progress": task_state.progress,
                },
            )
        )
    finally:
        if heartbeat_task is not None:
            with contextlib.suppress(RuntimeError):
                heartbeat_task.cancel()


@router.post("/start", response_model=IndexStartResponse)
async def start_index(request: IndexRequest) -> IndexStartResponse:
    """Start a new indexing job."""
    # Create task in registry
    task = task_registry.create()

    # Spawn background task (intentionally not awaited)
    # ruff: noqa: RUF006
    asyncio.create_task(_run_indexing(task, request))

    return IndexStartResponse(job_id=task.job_id, status=task.status.value)


@router.get("/status/{job_id}", response_model=IndexStatusResponse)
async def get_index_status(job_id: str) -> IndexStatusResponse:
    """Get the status of an indexing job."""
    task = task_registry.get(job_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")

    rows_indexed = task.progress.get("rows_indexed", 0)
    total_rows = task.progress.get("total_rows")
    errors = task.progress.get("errors", 0)

    return IndexStatusResponse(
        job_id=job_id,
        status=task.status.value,
        rows_indexed=rows_indexed,
        total_rows=total_rows,
        errors=errors,
        error=task.error,
    )


@router.post("/cancel/{job_id}", response_model=MessageResponse)
async def cancel_index(job_id: str) -> MessageResponse:
    """Cancel a running indexing job."""
    task = task_registry.get(job_id)

    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")

    success = task_registry.cancel(job_id)

    if not success:
        raise HTTPException(status_code=400, detail="Job is not cancellable")

    return MessageResponse(message=f"Job {job_id} cancelled")


@router.websocket("/ws/{job_id}")
async def index_ws(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for real-time indexing progress updates."""
    task = task_registry.get(job_id)

    if task is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await ws_manager.connect(job_id, websocket)

    try:
        while task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
            await asyncio.sleep(0.5)

        # Send final status
        await ws_manager.broadcast(
            job_id,
            {
                "status": task.status.value,
                "error": task.error,
                "progress": task.progress,
            },
        )
    except Exception:
        logger.warning("WebSocket disconnected for job %s", job_id)
    finally:
        await ws_manager.disconnect(job_id, websocket)
