from __future__ import annotations

import logging
import random
from typing import Any, cast

from fastapi import APIRouter, HTTPException

from embedding_cluster.ai_naming import (
    get_cluster_name,
    get_sub_cluster_name,
)
from embedding_cluster.ai_naming import (
    test_connection as ai_test_connection,
)
from embedding_cluster.server.models import (
    AiNamingRequest,
    AiNamingResponse,
    AiSubClusterNamingRequest,
    AiTestConnectionRequest,
    AiTestConnectionResponse,
)
from embedding_cluster.server.tasks import TaskStatus, task_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["ai"])

MAX_SAMPLE_ITEMS = 10


def _get_completed_job(job_id: str) -> dict[str, Any]:
    task = task_registry.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job not completed")
    return cast("dict[str, Any]", task.result)


def _get_item_names_for_cluster(
    result: dict[str, Any],
    cluster_index: int,
) -> list[str]:
    points = cast("list[dict[str, Any]]", result["points"])
    cluster_labels = cast("list[int]", result["cluster_labels"])

    cluster_point_indices = [
        i for i, label in enumerate(cluster_labels) if label == cluster_index
    ]

    sample_indices = random.sample(
        cluster_point_indices,
        min(MAX_SAMPLE_ITEMS, len(cluster_point_indices)),
    )

    names: list[str] = []
    for idx in sample_indices:
        point = points[idx]
        metadata = cast("dict[str, Any]", point.get("metadata", {}))
        name_parts = [str(v) for v in metadata.values()]
        names.append(
            ", ".join(name_parts) if name_parts else f"Item {idx}",
        )

    return names


def _get_item_names_for_sub_cluster(
    result: dict[str, Any],
    point_ids: list[str],
    sub_cluster_labels: list[int],
    sub_cluster_index: int,
) -> list[str]:
    points = cast("list[dict[str, Any]]", result["points"])
    point_id_to_point = {cast("str", p["id"]): p for p in points}

    sub_indices = [
        i for i, label in enumerate(sub_cluster_labels) if label == sub_cluster_index
    ]

    sample_indices = random.sample(
        sub_indices,
        min(MAX_SAMPLE_ITEMS, len(sub_indices)),
    )

    names: list[str] = []
    for idx in sample_indices:
        pid = point_ids[idx]
        point = point_id_to_point.get(pid)
        if point:
            metadata = cast("dict[str, Any]", point.get("metadata", {}))
            name_parts = [str(v) for v in metadata.values()]
            names.append(
                ", ".join(name_parts) if name_parts else f"Item {pid}",
            )
        else:
            names.append(f"Item {pid}")

    return names


@router.post("/name-clusters", response_model=AiNamingResponse)
async def name_clusters(request: AiNamingRequest) -> AiNamingResponse:
    result = _get_completed_job(request.job_id)

    names: dict[str, str] = {}
    for cluster_index in request.cluster_indices:
        item_names = _get_item_names_for_cluster(result, cluster_index)
        name = get_cluster_name(
            item_names=item_names,
            api_key=request.api_key,
            model=request.model,
            base_url=request.base_url,
            temperature=request.temperature,
        )
        names[str(cluster_index)] = name

    return AiNamingResponse(names=names)


@router.post("/name-sub-clusters", response_model=AiNamingResponse)
async def name_sub_clusters(
    request: AiSubClusterNamingRequest,
) -> AiNamingResponse:
    result = _get_completed_job(request.job_id)

    unique_labels = sorted(set(request.sub_cluster_labels))
    names: dict[str, str] = {}

    for label in unique_labels:
        item_names = _get_item_names_for_sub_cluster(
            result,
            request.point_ids,
            request.sub_cluster_labels,
            label,
        )
        name = get_sub_cluster_name(
            item_names=item_names,
            api_key=request.api_key,
            model=request.model,
            base_url=request.base_url,
            temperature=request.temperature,
            parent_cluster_name=request.parent_cluster_name,
        )
        names[str(label)] = name

    return AiNamingResponse(names=names)


@router.post("/test-connection", response_model=AiTestConnectionResponse)
async def test_connection(
    request: AiTestConnectionRequest,
) -> AiTestConnectionResponse:
    success, error = ai_test_connection(
        api_key=request.api_key,
        model=request.model,
        base_url=request.base_url,
    )
    return AiTestConnectionResponse(success=success, error=error)
