from __future__ import annotations

import asyncio
import logging
from typing import cast

from fastapi import APIRouter, HTTPException

from embedding_cluster.scatter_plot import (
    compute_plot_data,
    load_chromadb_embeddings,
    suggest_optimal_clusters,
)
from embedding_cluster.server.models import (
    IndexStartResponse,
    PlotRequest,
    SuggestClustersRequest,
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
            reduction_algorithm=request.reduction_algorithm,
            tsne_perplexity=request.tsne_perplexity,
            tsne_learning_rate=request.tsne_learning_rate,
            umap_n_neighbors=request.umap_n_neighbors,
            umap_min_dist=request.umap_min_dist,
            umap_metric=request.umap_metric,
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
    _ = asyncio.create_task(_run_compute(task, request))  # noqa: RUF006
    return IndexStartResponse(job_id=task.job_id, status=task.status.value)


@router.get("/data/{job_id}")
async def get_plot_data(job_id: str) -> dict[str, object]:
    task = task_registry.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        return {"status": task.status.value, "ready": False}
    if task.status == TaskStatus.FAILED:
        return {"status": "failed", "error": task.error, "ready": False}
    # COMPLETED
    result = cast("dict[str, object]", task.result)
    return {
        "status": "completed",
        "ready": True,
        **result,
    }


async def _run_suggest_clusters(
    task_state: TaskState, request: SuggestClustersRequest
) -> None:
    try:
        task_state.status = TaskStatus.RUNNING
        task_state.progress = {"phase": "loading_embeddings"}
        embeddings = await asyncio.to_thread(
            load_chromadb_embeddings, request.collection_name
        )

        def on_progress(info: dict[str, object]) -> None:
            task_state.progress = {**task_state.progress, **info}

        task_state.progress = {
            "phase": "analyzing",
            "current_k": 0,
            "total_k": 0,
        }
        result = await asyncio.to_thread(
            suggest_optimal_clusters,
            embeddings,
            k_range=range(request.k_min, request.k_max),
            max_samples=5000,
            on_progress=on_progress,
        )
        task_state.result = result
        task_state.status = TaskStatus.COMPLETED
    except Exception as e:
        logger.exception("Suggest-clusters failed for job %s", task_state.job_id)
        task_state.status = TaskStatus.FAILED
        task_state.error = str(e)


@router.post("/suggest-clusters", response_model=IndexStartResponse)
async def suggest_clusters(
    request: SuggestClustersRequest,
) -> IndexStartResponse:
    task = task_registry.create()
    _ = asyncio.create_task(_run_suggest_clusters(task, request))  # noqa: RUF006
    return IndexStartResponse(job_id=task.job_id, status=task.status.value)


@router.get("/suggest-clusters/{job_id}")
async def get_suggest_clusters_status(job_id: str) -> dict[str, object]:
    task = task_registry.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
        return {
            "status": task.status.value,
            "ready": False,
            "phase": task.progress.get("phase"),
            "current_k": task.progress.get("current_k"),
            "total_k": task.progress.get("total_k"),
        }
    if task.status == TaskStatus.FAILED:
        return {
            "status": "failed",
            "ready": False,
            "error": task.error,
        }
    result = cast("dict[str, object]", task.result)
    return {
        "status": "completed",
        "ready": True,
        "result": result,
    }
