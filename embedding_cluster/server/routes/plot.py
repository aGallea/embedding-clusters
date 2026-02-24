from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from embedding_cluster.scatter_plot import compute_plot_data
from embedding_cluster.server.models import (
    IndexStartResponse,
    PlotRequest,
)
from embedding_cluster.server.tasks import TaskState, TaskStatus, task_registry
from embedding_cluster.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plot", tags=["plot"])


async def _run_compute(task_state: TaskState, request: PlotRequest) -> None:
    """Run plot computation in background."""
    try:
        settings = Settings(
            running_mode="PLOT",
            chromadb_collection_name=request.chromadb_collection_name,
            num_clusters=request.num_clusters,
            text_display_fields=request.text_display_fields,
            image_field=request.image_field,
            gpt_generate_cluster_name=request.gpt_generate_cluster_name,
            gpt_default_model=request.gpt_default_model,
            gpt_default_temperature=request.gpt_default_temperature,
        )
        task_state.status = TaskStatus.RUNNING
        result = await asyncio.to_thread(compute_plot_data, settings)
        task_state.result = result
        task_state.status = TaskStatus.COMPLETED
    except Exception as e:
        logger.exception("Plot compute failed for job %s", task_state.job_id)
        task_state.status = TaskStatus.FAILED
        task_state.error = str(e)


@router.post("/compute", response_model=IndexStartResponse)
async def start_compute(request: PlotRequest) -> IndexStartResponse:
    task = task_registry.create()
    asyncio.create_task(_run_compute(task, request))  # noqa: RUF006
    return IndexStartResponse(job_id=task.job_id, status=task.status.value)


@router.get("/data/{job_id}")
async def get_plot_data(job_id: str) -> dict[str, Any]:
    task = task_registry.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        return {"status": task.status.value, "ready": False}
    if task.status == TaskStatus.FAILED:
        return {"status": "failed", "error": task.error, "ready": False}
    # COMPLETED
    return {
        "status": "completed",
        "ready": True,
        **task.result,
    }
