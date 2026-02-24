from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    """Mock ChromaDB client for testing."""
    return MagicMock()


async def test_list_collections(client, mock_chromadb_client):
    """Test GET /api/collections returns list of collections with counts."""
    mock_collection1 = MagicMock()
    mock_collection1.count.return_value = 10

    mock_collection2 = MagicMock()
    mock_collection2.count.return_value = 25

    mock_chromadb_client.list_collections.return_value = [
        "collection1",
        "collection2",
    ]
    mock_chromadb_client.get_collection.side_effect = [
        mock_collection1,
        mock_collection2,
    ]

    with patch(
        "embedding_cluster.server.routes.collections._get_chromadb_client",
        return_value=mock_chromadb_client,
    ):
        response = await client.get("/api/collections")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0] == {"name": "collection1", "count": 10}
    assert data[1] == {"name": "collection2", "count": 25}


async def test_get_collection_success(client, mock_chromadb_client):
    """Test GET /api/collections/{name} returns collection detail."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 42
    mock_collection.peek.return_value = {
        "metadatas": [{"field1": "value1", "field2": "value2"}]
    }

    mock_chromadb_client.get_collection.return_value = mock_collection

    with patch(
        "embedding_cluster.server.routes.collections._get_chromadb_client",
        return_value=mock_chromadb_client,
    ):
        response = await client.get("/api/collections/test_collection")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_collection"
    assert data["count"] == 42
    assert data["metadata_fields"] == ["field1", "field2"]


async def test_get_collection_not_found(client, mock_chromadb_client):
    """Test GET /api/collections/{name} returns 404 for non-existent."""
    mock_chromadb_client.get_collection.side_effect = Exception("Collection not found")

    with patch(
        "embedding_cluster.server.routes.collections._get_chromadb_client",
        return_value=mock_chromadb_client,
    ):
        response = await client.get("/api/collections/nonexistent")

    assert response.status_code == 404
    data = response.json()
    assert "Collection not found" in data["detail"]


async def test_get_collection_empty_metadata(client, mock_chromadb_client):
    """Test GET /api/collections/{name} with empty collection."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    mock_chromadb_client.get_collection.return_value = mock_collection

    with patch(
        "embedding_cluster.server.routes.collections._get_chromadb_client",
        return_value=mock_chromadb_client,
    ):
        response = await client.get("/api/collections/empty_collection")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "empty_collection"
    assert data["count"] == 0
    assert data["metadata_fields"] == []


async def test_delete_collection_success(client, mock_chromadb_client):
    """Test DELETE /api/collections/{name} deletes successfully."""
    mock_chromadb_client.delete_collection.return_value = None

    with patch(
        "embedding_cluster.server.routes.collections._get_chromadb_client",
        return_value=mock_chromadb_client,
    ):
        response = await client.delete("/api/collections/test_collection")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Deleted collection: test_collection"
    mock_chromadb_client.delete_collection.assert_called_once_with("test_collection")


async def test_delete_collection_not_found(client, mock_chromadb_client):
    """Test DELETE /api/collections/{name} returns 404 for non-existent."""
    mock_chromadb_client.delete_collection.side_effect = Exception("Collection not found")

    with patch(
        "embedding_cluster.server.routes.collections._get_chromadb_client",
        return_value=mock_chromadb_client,
    ):
        response = await client.delete("/api/collections/nonexistent")

    assert response.status_code == 404
    data = response.json()
    assert "Collection not found" in data["detail"]
