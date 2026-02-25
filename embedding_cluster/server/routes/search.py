from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import chromadb
import torch
from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from embedding_cluster.server.models import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from embedding_cluster.utils import ImageDownloader

if TYPE_CHECKING:
    from chromadb.api import ClientAPI

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])

# Lazy-loaded model cache
_model_cache: dict[str, Any] = {}


def _get_chromadb_client() -> ClientAPI:
    return chromadb.PersistentClient(path="./chromadb")


def _get_text_model(model_name: str) -> SentenceTransformer:
    cache_key = f"text:{model_name}"
    if cache_key not in _model_cache:
        logger.info("Loading text model: %s", model_name)
        _model_cache[cache_key] = SentenceTransformer(model_name)
    return _model_cache[cache_key]  # type: ignore[no-any-return]


def _get_image_model(
    model_name: str,
) -> tuple[CLIPModel, CLIPProcessor]:
    cache_key = f"image:{model_name}"
    if cache_key not in _model_cache:
        logger.info("Loading image model: %s", model_name)
        _model_cache[cache_key] = (
            CLIPModel.from_pretrained(model_name),
            CLIPProcessor.from_pretrained(model_name),
        )
    return _model_cache[cache_key]  # type: ignore[no-any-return]


async def _generate_text_embedding(query_text: str, model_name: str) -> list[float]:
    model = _get_text_model(model_name)
    embedding = model.encode(query_text, show_progress_bar=False)
    return embedding.tolist()


async def _generate_image_embedding(image_url: str, model_name: str) -> list[float]:
    image = await ImageDownloader().download_image_exp_backoff(image_url)
    if image is None:
        msg = f"Failed to download image from {image_url}"
        raise ValueError(msg)

    model, processor = _get_image_model(model_name)
    inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        img_features = model.get_image_features(inputs["pixel_values"])
    return img_features.squeeze(0).cpu().numpy().tolist()  # type: ignore[no-any-return]


async def _generate_clip_text_embedding(query_text: str, model_name: str) -> list[float]:
    """Encode text using CLIP's text encoder.

    This produces embeddings in the same vector space as CLIP image
    embeddings, enabling text-to-image similarity search.
    """
    model, processor = _get_image_model(model_name)
    inputs = processor(text=query_text, images=None, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(inputs["input_ids"])
    return text_features.squeeze(0).cpu().numpy().tolist()  # type: ignore[no-any-return]


@router.post("", response_model=SearchResponse)
async def search_collection(
    request: SearchRequest,
) -> SearchResponse:
    if not request.query_text and not request.query_image_url:
        raise HTTPException(
            status_code=400,
            detail="Either query_text or query_image_url is required",
        )

    client = _get_chromadb_client()
    try:
        collection = client.get_collection(request.collection_name)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=(f"Collection not found: {request.collection_name}"),
        ) from e

    if collection.count() == 0:
        return SearchResponse(results=[])

    # Determine model from collection metadata
    metadata = collection.metadata or {}
    stored_model_name = metadata.get("model_name")
    stored_model_type = metadata.get("model_type")

    # Generate embedding
    try:
        if request.query_text:
            if stored_model_type == "image" and stored_model_name:
                # Collection was indexed with CLIP — use CLIP text encoder
                embedding = await _generate_clip_text_embedding(
                    request.query_text, stored_model_name
                )
            else:
                # Text collection or no metadata — use SentenceTransformer
                model_name = (
                    stored_model_name
                    if stored_model_name and stored_model_type == "text"
                    else request.text_model_name
                )
                embedding = await _generate_text_embedding(request.query_text, model_name)
        else:
            assert request.query_image_url is not None
            model_name = (
                stored_model_name
                if stored_model_name and stored_model_type == "image"
                else request.image_model_name
            )
            embedding = await _generate_image_embedding(
                request.query_image_url, model_name
            )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    # Query ChromaDB
    query_result = collection.query(
        query_embeddings=[embedding],  # type: ignore[arg-type]
        n_results=min(request.n_results, collection.count()),
    )

    results: list[SearchResult] = []
    if query_result["ids"] and query_result["distances"]:
        ids = query_result["ids"][0]
        distances = query_result["distances"][0]
        metadatas = (
            query_result["metadatas"][0] if query_result["metadatas"] else [{}] * len(ids)
        )
        for i, doc_id in enumerate(ids):
            results.append(
                SearchResult(
                    id=doc_id,
                    distance=distances[i],
                    metadata=dict(metadatas[i]) if metadatas[i] else {},
                )
            )

    return SearchResponse(results=results)
