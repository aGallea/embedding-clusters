from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from embedding_cluster.server.app import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_chromadb_client():
    mock_client = MagicMock()
    return mock_client


@pytest.fixture
def mock_collection(mock_chromadb_client):
    mock_coll = MagicMock()
    mock_coll.count.return_value = 100
    mock_coll.metadata = {
        "model_name": "BAAI/bge-small-en-v1.5",
        "model_type": "text",
    }
    mock_coll.query.return_value = {
        "ids": [["id1", "id2", "id3"]],
        "distances": [[0.1, 0.3, 0.5]],
        "metadatas": [
            [
                {
                    "name": "item1",
                    "imageUrl": "http://example.com/1.jpg",
                },
                {
                    "name": "item2",
                    "imageUrl": "http://example.com/2.jpg",
                },
                {
                    "name": "item3",
                    "imageUrl": "http://example.com/3.jpg",
                },
            ]
        ],
    }
    mock_chromadb_client.get_collection.return_value = mock_coll
    return mock_coll


@pytest.fixture
def mock_text_model():
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384)
    return mock_model


@pytest.fixture
def mock_image_model():
    mock_clip = MagicMock()
    mock_processor = MagicMock()

    import torch

    mock_features = torch.zeros(1, 512)
    mock_clip.get_image_features.return_value = mock_features

    return mock_clip, mock_processor


async def test_search_text_query(
    client, mock_chromadb_client, mock_collection, mock_text_model
):
    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_text_model",
            return_value=mock_text_model,
        ),
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_text": "red shoes",
                "n_results": 3,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3
    assert data["results"][0]["id"] == "id1"
    assert data["results"][0]["distance"] == 0.1
    assert "metadata" in data["results"][0]


async def test_search_image_query(
    client, mock_chromadb_client, mock_collection, mock_image_model
):
    mock_clip, mock_processor = mock_image_model
    mock_image = MagicMock()

    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_image_model",
            return_value=(mock_clip, mock_processor),
        ),
        patch(
            "embedding_cluster.server.routes.search.ImageDownloader"
        ) as mock_downloader_cls,
    ):
        mock_instance = MagicMock()
        mock_instance.download_image_exp_backoff = AsyncMock(return_value=mock_image)
        mock_downloader_cls.return_value = mock_instance

        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_image_url": "http://example.com/query.jpg",
                "model_type": "image",
                "n_results": 3,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 3


async def test_search_collection_not_found(client):
    mock_client = MagicMock()
    mock_client.get_collection.side_effect = Exception("Not found")

    with patch(
        "embedding_cluster.server.routes.search._get_chromadb_client",
        return_value=mock_client,
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "nonexistent",
                "query_text": "test",
            },
        )

    assert response.status_code == 404
    assert "Collection not found" in response.json()["detail"]


async def test_search_empty_query(client):
    response = await client.post(
        "/api/search",
        json={"collection_name": "test_collection"},
    )
    assert response.status_code == 400


async def test_search_custom_n_results(
    client, mock_chromadb_client, mock_collection, mock_text_model
):
    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_text_model",
            return_value=mock_text_model,
        ),
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_text": "blue jacket",
                "n_results": 5,
            },
        )

    assert response.status_code == 200
    mock_collection.query.assert_called_once()
    call_kwargs = mock_collection.query.call_args
    assert (
        call_kwargs.kwargs.get("n_results") == 5 or call_kwargs[1].get("n_results") == 5
    )


async def test_search_response_structure(
    client, mock_chromadb_client, mock_collection, mock_text_model
):
    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_text_model",
            return_value=mock_text_model,
        ),
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_text": "test",
            },
        )

    data = response.json()
    assert "results" in data
    for result in data["results"]:
        assert "id" in result
        assert "distance" in result
        assert "metadata" in result
        assert isinstance(result["distance"], float)


async def test_search_empty_collection(client, mock_chromadb_client):
    mock_coll = MagicMock()
    mock_coll.count.return_value = 0
    mock_chromadb_client.get_collection.return_value = mock_coll

    with patch(
        "embedding_cluster.server.routes.search._get_chromadb_client",
        return_value=mock_chromadb_client,
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "empty_collection",
                "query_text": "test",
            },
        )

    assert response.status_code == 200
    assert response.json()["results"] == []


async def test_search_image_download_failure(
    client, mock_chromadb_client, mock_collection
):
    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search.ImageDownloader"
        ) as mock_downloader_cls,
    ):
        mock_instance = MagicMock()
        mock_instance.download_image_exp_backoff = AsyncMock(return_value=None)
        mock_downloader_cls.return_value = mock_instance

        response = await client.post(
            "/api/search",
            json={
                "collection_name": "test_collection",
                "query_image_url": "http://example.com/bad.jpg",
                "model_type": "image",
            },
        )

    assert response.status_code == 500


async def test_search_text_on_image_collection_uses_clip(client, mock_chromadb_client):
    """Text query on an image (CLIP) collection should use CLIP text encoder."""
    import torch

    mock_coll = MagicMock()
    mock_coll.count.return_value = 100
    mock_coll.metadata = {
        "model_name": "openai/clip-vit-base-patch32",
        "model_type": "image",
    }
    mock_coll.query.return_value = {
        "ids": [["id1"]],
        "distances": [[0.2]],
        "metadatas": [[{"name": "item1"}]],
    }
    mock_chromadb_client.get_collection.return_value = mock_coll

    mock_clip = MagicMock()
    mock_processor = MagicMock()
    mock_text_features = torch.zeros(1, 512)
    mock_clip.get_text_features.return_value = mock_text_features

    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_image_model",
            return_value=(mock_clip, mock_processor),
        ),
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "clip_collection",
                "query_text": "red shoes",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    # Verify CLIP text encoder was used (not SentenceTransformer)
    mock_clip.get_text_features.assert_called_once()


async def test_search_uses_stored_text_model_name(client, mock_chromadb_client):
    """Text query should use the model name stored in collection metadata."""
    mock_coll = MagicMock()
    mock_coll.count.return_value = 100
    mock_coll.metadata = {
        "model_name": "custom-text-model",
        "model_type": "text",
    }
    mock_coll.query.return_value = {
        "ids": [["id1"]],
        "distances": [[0.1]],
        "metadatas": [[{"name": "item1"}]],
    }
    mock_chromadb_client.get_collection.return_value = mock_coll

    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384)

    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_text_model",
            return_value=mock_model,
        ) as mock_get_text,
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "text_collection",
                "query_text": "test query",
            },
        )

    assert response.status_code == 200
    # Verify the stored model name was used
    mock_get_text.assert_called_with("custom-text-model")


async def test_search_uses_stored_image_model_name(
    client, mock_chromadb_client, mock_image_model
):
    """Image query should use the model name stored in collection metadata."""
    mock_clip, mock_processor = mock_image_model
    mock_image = MagicMock()

    mock_coll = MagicMock()
    mock_coll.count.return_value = 100
    mock_coll.metadata = {
        "model_name": "custom-clip-model",
        "model_type": "image",
    }
    mock_coll.query.return_value = {
        "ids": [["id1"]],
        "distances": [[0.1]],
        "metadatas": [[{"name": "item1"}]],
    }
    mock_chromadb_client.get_collection.return_value = mock_coll

    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_image_model",
            return_value=(mock_clip, mock_processor),
        ) as mock_get_image,
        patch(
            "embedding_cluster.server.routes.search.ImageDownloader"
        ) as mock_downloader_cls,
    ):
        mock_instance = MagicMock()
        mock_instance.download_image_exp_backoff = AsyncMock(return_value=mock_image)
        mock_downloader_cls.return_value = mock_instance

        response = await client.post(
            "/api/search",
            json={
                "collection_name": "image_collection",
                "query_image_url": "http://example.com/test.jpg",
            },
        )

    assert response.status_code == 200
    mock_get_image.assert_called_with("custom-clip-model")


async def test_search_fallback_when_no_metadata(
    client, mock_chromadb_client, mock_text_model
):
    """When collection has no metadata, fall back to request defaults."""
    mock_coll = MagicMock()
    mock_coll.count.return_value = 100
    mock_coll.metadata = None
    mock_coll.query.return_value = {
        "ids": [["id1"]],
        "distances": [[0.1]],
        "metadatas": [[{"name": "item1"}]],
    }
    mock_chromadb_client.get_collection.return_value = mock_coll

    with (
        patch(
            "embedding_cluster.server.routes.search._get_chromadb_client",
            return_value=mock_chromadb_client,
        ),
        patch(
            "embedding_cluster.server.routes.search._get_text_model",
            return_value=mock_text_model,
        ) as mock_get_text,
    ):
        response = await client.post(
            "/api/search",
            json={
                "collection_name": "legacy_collection",
                "query_text": "test",
            },
        )

    assert response.status_code == 200
    # Should fall back to default text model name from request
    mock_get_text.assert_called_with("BAAI/bge-small-en-v1.5")


def test_get_chromadb_client():
    """Test _get_chromadb_client() calls chromadb.PersistentClient with correct path."""
    from embedding_cluster.server.routes.search import _get_chromadb_client

    with patch(
        "embedding_cluster.server.routes.search.chromadb.PersistentClient"
    ) as mock_persistent:
        mock_client = MagicMock()
        mock_persistent.return_value = mock_client

        result = _get_chromadb_client()

        # Verify PersistentClient was called with correct path
        mock_persistent.assert_called_once_with(path="./chromadb")
        # Verify return value is the mock client
        assert result == mock_client


def test_get_text_model_cache_miss():
    """Test _get_text_model() loads SentenceTransformer on cache miss."""
    from embedding_cluster.server.routes.search import (
        _get_text_model,
        _model_cache,
    )

    # Clear cache before test
    _model_cache.clear()

    with patch(
        "embedding_cluster.server.routes.search.SentenceTransformer"
    ) as mock_sentence_transformer:
        mock_model_instance = MagicMock()
        mock_sentence_transformer.return_value = mock_model_instance

        result = _get_text_model("test-model")

        # Verify SentenceTransformer was called with model name
        mock_sentence_transformer.assert_called_once_with("test-model")
        # Verify return value is the mock instance
        assert result == mock_model_instance
        # Verify cache now contains the key
        assert "text:test-model" in _model_cache
        assert _model_cache["text:test-model"] == mock_model_instance

    # Clean up
    _model_cache.clear()


def test_get_text_model_cache_hit():
    """Test _get_text_model() uses cached model on cache hit."""
    from embedding_cluster.server.routes.search import (
        _get_text_model,
        _model_cache,
    )

    # Clear cache and set up cached value
    _model_cache.clear()
    mock_cached_model = MagicMock()
    _model_cache["text:cached-model"] = mock_cached_model

    with patch(
        "embedding_cluster.server.routes.search.SentenceTransformer"
    ) as mock_sentence_transformer:
        result = _get_text_model("cached-model")

        # Verify SentenceTransformer was NOT called (cache hit)
        mock_sentence_transformer.assert_not_called()
        # Verify return value is the cached instance
        assert result == mock_cached_model

    # Clean up
    _model_cache.clear()


def test_get_image_model_cache_miss():
    """Test _get_image_model() loads CLIPModel and CLIPProcessor on cache miss."""
    from embedding_cluster.server.routes.search import (
        _get_image_model,
        _model_cache,
    )

    # Clear cache before test
    _model_cache.clear()

    with (
        patch(
            "embedding_cluster.server.routes.search.CLIPModel.from_pretrained"
        ) as mock_clip_from_pretrained,
        patch(
            "embedding_cluster.server.routes.search.CLIPProcessor.from_pretrained"
        ) as mock_processor_from_pretrained,
    ):
        mock_clip_instance = MagicMock()
        mock_processor_instance = MagicMock()
        mock_clip_from_pretrained.return_value = mock_clip_instance
        mock_processor_from_pretrained.return_value = mock_processor_instance

        result = _get_image_model("test-clip")

        # Verify both from_pretrained were called with model name
        mock_clip_from_pretrained.assert_called_once_with("test-clip")
        mock_processor_from_pretrained.assert_called_once_with("test-clip")
        # Verify return value is a tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == mock_clip_instance
        assert result[1] == mock_processor_instance
        # Verify cache now contains the key
        assert "image:test-clip" in _model_cache
        assert _model_cache["image:test-clip"] == result

    # Clean up
    _model_cache.clear()


def test_get_image_model_cache_hit():
    """Test _get_image_model() uses cached model on cache hit."""
    from embedding_cluster.server.routes.search import (
        _get_image_model,
        _model_cache,
    )

    # Clear cache and set up cached value
    _model_cache.clear()
    mock_clip = MagicMock()
    mock_processor = MagicMock()
    mock_tuple = (mock_clip, mock_processor)
    _model_cache["image:cached-clip"] = mock_tuple

    with (
        patch(
            "embedding_cluster.server.routes.search.CLIPModel.from_pretrained"
        ) as mock_clip_from_pretrained,
        patch(
            "embedding_cluster.server.routes.search.CLIPProcessor.from_pretrained"
        ) as mock_processor_from_pretrained,
    ):
        result = _get_image_model("cached-clip")

        # Verify from_pretrained were NOT called (cache hit)
        mock_clip_from_pretrained.assert_not_called()
        mock_processor_from_pretrained.assert_not_called()
        # Verify return value is the cached tuple
        assert result == mock_tuple

    # Clean up
    _model_cache.clear()
