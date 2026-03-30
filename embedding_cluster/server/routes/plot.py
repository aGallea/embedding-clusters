from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from fastapi import APIRouter, HTTPException

from embedding_cluster.scatter_plot import (
    compute_plot_data,
    load_chromadb_embeddings,
    reduce_dimensions,
    suggest_optimal_clusters,
)
from embedding_cluster.server.models import (
    ClusterDetailResponse,
    ClusterItemResponse,
    IndexStartResponse,
    PlotRequest,
    SubClusterInfo,
    SubClusterPoint,
    SubClusterRequest,
    SubClusterResponse,
    SuggestClustersRequest,
    SuggestKRequest,
    SuggestKResponse,
    SuggestKScoreEntry,
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
    # Strip internal fields not meant for the frontend
    internal_keys = ("embeddings_standardized", "cluster_labels", "point_ids")
    frontend_result = {k: v for k, v in result.items() if k not in internal_keys}
    return {
        "status": "completed",
        "ready": True,
        "job_id": job_id,
        **frontend_result,
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


@router.get(
    "/{job_id}/cluster/{cluster_index}",
    response_model=ClusterDetailResponse,
)
async def get_cluster_detail(
    job_id: str,
    cluster_index: int,
    page: int = 1,
    page_size: int = 50,
) -> ClusterDetailResponse:
    task = task_registry.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")

    result = cast("dict[str, object]", task.result)
    clusters = cast("list[dict[str, object]]", result["clusters"])
    cluster_labels = cast("list[int]", result["cluster_labels"])
    embeddings = cast("list[list[float]]", result["embeddings_standardized"])
    points = cast("list[dict[str, object]]", result["points"])

    # Validate cluster index
    cluster_info: dict[str, Any] | None = None
    for c in clusters:
        if cast("int", c["index"]) == cluster_index:
            cluster_info = cast("dict[str, Any]", c)
            break
    if cluster_info is None:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # Get indices belonging to this cluster
    cluster_point_indices = [
        i for i, label in enumerate(cluster_labels) if label == cluster_index
    ]

    # Compute centroid
    import numpy as np

    cluster_embeddings = np.array([embeddings[i] for i in cluster_point_indices])
    centroid = cluster_embeddings.mean(axis=0)

    # Compute distances and build items
    items_with_distance: list[tuple[float, dict[str, object]]] = []
    for idx in cluster_point_indices:
        point_embedding = np.array(embeddings[idx])
        distance = float(np.linalg.norm(point_embedding - centroid))
        point = points[idx]
        items_with_distance.append((distance, point))

    # Sort by distance
    items_with_distance.sort(key=lambda x: x[0])

    # Paginate
    total_items = len(items_with_distance)
    start = (page - 1) * page_size
    end = start + page_size
    page_items = items_with_distance[start:end]

    return ClusterDetailResponse(
        cluster_index=cluster_index,
        cluster_name=cast("str", cluster_info["name"]),
        total_items=total_items,
        page=page,
        page_size=page_size,
        items=[
            ClusterItemResponse(
                id=cast("str", point["id"]),
                metadata=cast("dict[str, object]", point["metadata"]),
                distance_to_centroid=dist,
            )
            for dist, point in page_items
        ],
    )


@router.post(
    "/{job_id}/sub-cluster",
    response_model=SubClusterResponse,
)
async def sub_cluster_generic(
    job_id: str,
    request: SubClusterRequest,
) -> SubClusterResponse:
    """Sub-cluster by point_ids (recursive)."""
    task = task_registry.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")

    result = cast("dict[str, object]", task.result)
    embeddings = cast("list[list[float]]", result["embeddings_standardized"])
    points = cast("list[dict[str, object]]", result["points"])
    point_ids_all = cast("list[str]", result["point_ids"])

    id_to_index = {pid: i for i, pid in enumerate(point_ids_all)}

    if request.point_ids is not None:
        cluster_point_indices = [
            id_to_index[pid] for pid in request.point_ids if pid in id_to_index
        ]
        if len(cluster_point_indices) == 0:
            raise HTTPException(
                status_code=400,
                detail="No valid point_ids found in job data",
            )
        parent_cluster_index = -1
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "point_ids is required for this endpoint. "
                "Use /{job_id}/cluster/{cluster_index}/sub-cluster "
                "for top-level cluster sub-clustering."
            ),
        )

    num_sub = request.num_sub_clusters
    if num_sub > len(cluster_point_indices):
        raise HTTPException(
            status_code=400,
            detail=(
                f"num_sub_clusters ({num_sub}) exceeds "
                f"points count ({len(cluster_point_indices)})"
            ),
        )

    import numpy as np

    cluster_embeddings = np.array([embeddings[i] for i in cluster_point_indices])

    def _compute_generic() -> SubClusterResponse:
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=num_sub,
            n_init="auto",
            random_state=171,
            max_iter=1000,
        )
        sub_labels = kmeans.fit_predict(cluster_embeddings)

        reduced = reduce_dimensions(
            cluster_embeddings,
            algorithm="pca",
            n_components=3,
        )

        sub_points: list[SubClusterPoint] = []
        for j, idx in enumerate(cluster_point_indices):
            point = points[idx]
            sub_points.append(
                SubClusterPoint(
                    id=cast("str", point["id"]),
                    x=float(reduced[j, 0]),
                    y=float(reduced[j, 1]),
                    z=float(reduced[j, 2]),
                    sub_cluster=int(sub_labels[j]),
                    metadata=cast(
                        "dict[str, object]",
                        point["metadata"],
                    ),
                )
            )

        sub_cluster_infos: list[SubClusterInfo] = []
        for si in range(num_sub):
            count = int(np.sum(sub_labels == si))
            color = f"hsl({si * 360 // num_sub}, 70%, 50%)"
            sub_cluster_infos.append(SubClusterInfo(index=si, count=count, color=color))

        return SubClusterResponse(
            parent_cluster_index=parent_cluster_index,
            points=sub_points,
            sub_clusters=sub_cluster_infos,
            total_points=len(cluster_point_indices),
        )

    return await asyncio.to_thread(_compute_generic)


@router.post(
    "/{job_id}/suggest-k",
    response_model=SuggestKResponse,
)
async def suggest_k(
    job_id: str,
    request: SuggestKRequest,
) -> SuggestKResponse:
    """Suggest optimal sub-cluster count via silhouette analysis."""
    task = task_registry.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")

    result = cast("dict[str, object]", task.result)
    embeddings = cast("list[list[float]]", result["embeddings_standardized"])
    point_ids_all = cast("list[str]", result["point_ids"])
    cluster_labels = cast("list[int]", result["cluster_labels"])

    id_to_index = {pid: i for i, pid in enumerate(point_ids_all)}

    if request.point_ids is not None:
        indices = [id_to_index[pid] for pid in request.point_ids if pid in id_to_index]
    elif request.cluster_index is not None:
        indices = [
            i for i, label in enumerate(cluster_labels) if label == request.cluster_index
        ]
    else:
        raise HTTPException(
            status_code=400,
            detail="Either point_ids or cluster_index is required",
        )

    if len(indices) < 3:
        raise HTTPException(
            status_code=400,
            detail="Need at least 3 points for silhouette analysis",
        )

    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as sk_silhouette_score
    from sklearn.preprocessing import StandardScaler

    subset_embeddings = np.array([embeddings[i] for i in indices])

    def _compute_suggest_k() -> SuggestKResponse:
        scaled = StandardScaler().fit_transform(subset_embeddings)
        max_k = min(request.max_k, len(indices) - 1)
        scores: list[SuggestKScoreEntry] = []
        best_k = 2
        best_score = -1.0

        for k in range(2, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                n_init="auto",
                random_state=171,
                max_iter=1000,
            )
            labels = kmeans.fit_predict(scaled)
            score = float(sk_silhouette_score(scaled, labels))
            scores.append(SuggestKScoreEntry(k=k, score=score))
            if score > best_score:
                best_score = score
                best_k = k

        return SuggestKResponse(
            suggested_k=best_k,
            scores=scores,
        )

    return await asyncio.to_thread(_compute_suggest_k)


@router.post(
    "/{job_id}/cluster/{cluster_index}/sub-cluster",
    response_model=SubClusterResponse,
)
async def sub_cluster(
    job_id: str,
    cluster_index: int,
    request: SubClusterRequest,
) -> SubClusterResponse:
    task = task_registry.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")

    result = cast("dict[str, object]", task.result)
    clusters = cast("list[dict[str, object]]", result["clusters"])
    cluster_labels = cast("list[int]", result["cluster_labels"])
    embeddings = cast("list[list[float]]", result["embeddings_standardized"])
    points = cast("list[dict[str, object]]", result["points"])

    # Validate cluster exists
    cluster_exists = any(cast("int", c["index"]) == cluster_index for c in clusters)
    if not cluster_exists:
        raise HTTPException(status_code=404, detail="Cluster not found")

    # Get indices for this cluster
    cluster_point_indices = [
        i for i, label in enumerate(cluster_labels) if label == cluster_index
    ]

    num_sub = request.num_sub_clusters
    if num_sub > len(cluster_point_indices):
        raise HTTPException(
            status_code=400,
            detail=(
                f"num_sub_clusters ({num_sub}) exceeds "
                f"items in cluster ({len(cluster_point_indices)})"
            ),
        )

    # Run k-means on cluster subset
    import numpy as np

    cluster_embeddings = np.array([embeddings[i] for i in cluster_point_indices])

    def _compute() -> SubClusterResponse:
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=num_sub,
            n_init="auto",
            random_state=171,
            max_iter=1000,
        )
        sub_labels = kmeans.fit_predict(cluster_embeddings)

        # Reduce dimensions for visualization
        reduced = reduce_dimensions(
            cluster_embeddings,
            algorithm="pca",
            n_components=3,
        )

        sub_points: list[SubClusterPoint] = []
        for j, idx in enumerate(cluster_point_indices):
            point = points[idx]
            sub_points.append(
                SubClusterPoint(
                    id=cast("str", point["id"]),
                    x=float(reduced[j, 0]),
                    y=float(reduced[j, 1]),
                    z=float(reduced[j, 2]),
                    sub_cluster=int(sub_labels[j]),
                    metadata=cast(
                        "dict[str, object]",
                        point["metadata"],
                    ),
                )
            )

        sub_cluster_infos: list[SubClusterInfo] = []
        for si in range(num_sub):
            count = int(np.sum(sub_labels == si))
            color = f"hsl({si * 360 // num_sub}, 70%, 50%)"
            sub_cluster_infos.append(SubClusterInfo(index=si, count=count, color=color))

        return SubClusterResponse(
            parent_cluster_index=cluster_index,
            points=sub_points,
            sub_clusters=sub_cluster_infos,
            total_points=len(cluster_point_indices),
        )

    return await asyncio.to_thread(_compute)
