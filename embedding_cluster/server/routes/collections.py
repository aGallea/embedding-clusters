from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb
from fastapi import APIRouter, HTTPException

from embedding_cluster.server.models import (
    CollectionDetail,
    CollectionInfo,
    MessageResponse,
)

if TYPE_CHECKING:
    from chromadb.api import ClientAPI

router = APIRouter(prefix="/api/collections", tags=["collections"])


def _get_chromadb_client() -> ClientAPI:
    return chromadb.PersistentClient(path="./chromadb")


@router.get("", response_model=list[CollectionInfo])
async def list_collections() -> list[CollectionInfo]:
    client = _get_chromadb_client()
    collection_names = client.list_collections()
    return [
        CollectionInfo(name=c, count=client.get_collection(c).count())
        for c in collection_names
    ]


@router.get("/{name}", response_model=CollectionDetail)
async def get_collection(name: str) -> CollectionDetail:
    client = _get_chromadb_client()
    try:
        collection = client.get_collection(name)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Collection not found: {name}"
        ) from e
    count = collection.count()
    metadata_fields: list[str] = []
    if count > 0:
        sample = collection.peek(limit=1)
        if sample["metadatas"] and len(sample["metadatas"]) > 0:
            metadata_fields = sorted(sample["metadatas"][0].keys())
    return CollectionDetail(name=name, count=count, metadata_fields=metadata_fields)


@router.delete("/{name}", response_model=MessageResponse)
async def delete_collection(name: str) -> MessageResponse:
    client = _get_chromadb_client()
    try:
        client.delete_collection(name)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Collection not found: {name}"
        ) from e
    return MessageResponse(message=f"Deleted collection: {name}")
