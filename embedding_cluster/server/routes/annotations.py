from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter

from embedding_cluster.annotations import AnnotationManager
from embedding_cluster.server.models import (
    AnnotationsResponse,
    AnnotationUpdate,
    MessageResponse,
)

logger = logging.getLogger(__name__)

_DEFAULT_ANNOTATIONS_DIR = Path("./annotations")

router = APIRouter(prefix="/api/annotations", tags=["annotations"])


def _get_manager() -> AnnotationManager:
    return AnnotationManager(base_dir=_DEFAULT_ANNOTATIONS_DIR)


@router.get("/{job_id}", response_model=AnnotationsResponse)
async def get_annotations(job_id: str) -> AnnotationsResponse:
    manager = _get_manager()
    data = manager.get_annotations(job_id)
    return AnnotationsResponse(**data)


@router.put(
    "/{job_id}/cluster/{cluster_index}",
    response_model=AnnotationsResponse,
)
async def update_annotation(
    job_id: str,
    cluster_index: int,
    body: AnnotationUpdate,
) -> AnnotationsResponse:
    manager = _get_manager()
    data = manager.update_annotation(
        job_id,
        cluster_index,
        name=body.name,
        notes=body.notes,
        tags=body.tags,
    )
    return AnnotationsResponse(**data)


@router.delete("/{job_id}", response_model=MessageResponse)
async def delete_annotations(job_id: str) -> MessageResponse:
    manager = _get_manager()
    manager.delete_annotations(job_id)
    return MessageResponse(message="Annotations deleted")
