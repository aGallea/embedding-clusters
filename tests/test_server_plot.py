from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from fastapi import FastAPI

from embedding_cluster.server.app import create_app


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
def mock_compute():
    def fake_compute(settings):
        return {
            "points": [
                {
                    "x": 1.0,
                    "y": 2.0,
                    "z": 3.0,
                    "cluster": 0,
                    "metadata": {"name": "item1"},
                    "id": "1",
                },
                {
                    "x": 4.0,
                    "y": 5.0,
                    "z": 6.0,
                    "cluster": 1,
                    "metadata": {"name": "item2"},
                    "id": "2",
                },
            ],
            "clusters": [
                {"index": 0, "name": "Group 1", "color": "hsl(0, 70%, 50%)", "count": 1},
                {
                    "index": 1,
                    "name": "Group 2",
                    "color": "hsl(180, 70%, 50%)",
                    "count": 1,
                },
            ],
            "total_points": 2,
        }

    with patch(
        "embedding_cluster.server.routes.plot.compute_plot_data", side_effect=fake_compute
    ):
        yield


@pytest.mark.asyncio
async def test_start_compute_success(app: FastAPI, mock_compute) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test_collection"},
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_start_compute_runs_in_thread(app: FastAPI) -> None:
    def fake_compute(settings):
        return {"points": [], "clusters": [], "total_points": 0}

    with (
        patch(
            "embedding_cluster.server.routes.plot.compute_plot_data",
            side_effect=fake_compute,
        ) as compute,
        patch("embedding_cluster.server.routes.plot.asyncio.to_thread") as to_thread,
    ):

        async def fake_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        to_thread.side_effect = fake_to_thread

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/plot/compute",
                json={"chromadb_collection_name": "test_collection"},
            )

        assert response.status_code == status.HTTP_200_OK
        await asyncio.sleep(0.1)

    to_thread.assert_called_once()
    compute.assert_called_once()


@pytest.mark.asyncio
async def test_start_compute_missing_collection(app: FastAPI, mock_compute) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/compute",
            json={},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_get_data_not_found(app: FastAPI, mock_compute) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/plot/data/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_get_data_pending(app: FastAPI, mock_compute) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start_response = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test_collection"},
        )
        job_id = start_response.json()["job_id"]

        # Immediately check status
        response = await client.get(f"/api/plot/data/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["ready"] is False
    assert data["status"] in ["pending", "running", "completed"]


@pytest.mark.asyncio
async def test_get_data_completed(app: FastAPI, mock_compute) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start_response = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test_collection"},
        )
        job_id = start_response.json()["job_id"]

        # Wait for completion
        await asyncio.sleep(0.2)

        response = await client.get(f"/api/plot/data/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["ready"] is True
    assert data["status"] == "completed"
    assert "points" in data
    assert "clusters" in data
    assert "total_points" in data
    assert len(data["points"]) == 2
    assert len(data["clusters"]) == 2
    assert data["total_points"] == 2


@pytest.mark.asyncio
async def test_get_data_failed(app: FastAPI) -> None:
    def failing_compute(settings):
        raise ValueError("Computation failed")

    with patch(
        "embedding_cluster.server.routes.plot.compute_plot_data",
        side_effect=failing_compute,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            start_response = await client.post(
                "/api/plot/compute",
                json={"chromadb_collection_name": "test_collection"},
            )
            job_id = start_response.json()["job_id"]

            # Wait for computation to fail
            await asyncio.sleep(0.2)

            response = await client.get(f"/api/plot/data/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["ready"] is False
    assert data["status"] == "failed"
    assert "error" in data
    assert "Computation failed" in data["error"]


@pytest.mark.asyncio
async def test_compute_with_all_fields(app: FastAPI, mock_compute) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/compute",
            json={
                "chromadb_collection_name": "test_collection",
                "num_clusters": 5,
                "text_display_fields": ["name", "description"],
                "image_field": "imageUrl",
                "gpt_generate_cluster_name": True,
                "gpt_default_model": "gpt-4",
                "gpt_default_temperature": 0.7,
            },
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"


@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB client to return a collection with embeddings."""
    mock_embeddings = np.random.default_rng(42).random((20, 10)).tolist()
    mock_collection = MagicMock()
    mock_collection.get.return_value = {"embeddings": mock_embeddings}
    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_collection
    with patch(
        "embedding_cluster.scatter_plot.chromadb.PersistentClient",
        return_value=mock_client,
    ):
        yield


@pytest.fixture
def mock_suggest():
    """Mock suggest_optimal_clusters to return deterministic results."""
    _default_k_range = range(2, 31)

    def fake_suggest(embeddings, k_range=None, max_samples=5000):
        if k_range is None:
            k_range = _default_k_range
        k_values = list(k_range)
        return {
            "k_values": k_values,
            "inertias": [100.0 / k for k in k_values],
            "silhouette_scores": [0.5 + 0.01 * k for k in k_values],
            "suggested_k": max(k_range),
        }

    with patch(
        "embedding_cluster.server.routes.plot.suggest_optimal_clusters",
        side_effect=fake_suggest,
    ):
        yield


@pytest.mark.asyncio
async def test_suggest_clusters_success(
    app: FastAPI, mock_chromadb_collection, mock_suggest
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/suggest-clusters",
            json={"collection_name": "test_collection"},
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "k_values" in data
    assert "inertias" in data
    assert "silhouette_scores" in data
    assert "suggested_k" in data
    assert data["k_values"][0] == 2
    assert data["k_values"][-1] == 29


@pytest.mark.asyncio
async def test_suggest_clusters_custom_range(
    app: FastAPI, mock_chromadb_collection, mock_suggest
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/suggest-clusters",
            json={"collection_name": "test_collection", "k_min": 3, "k_max": 15},
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    k_values = data["k_values"]
    assert k_values[0] == 3
    assert k_values[-1] == 14


@pytest.mark.asyncio
async def test_suggest_clusters_missing_collection(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/suggest-clusters",
            json={},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_suggest_clusters_collection_not_found(app: FastAPI) -> None:
    with patch(
        "embedding_cluster.server.routes.plot.load_chromadb_embeddings",
        side_effect=ValueError("Collection not_found does not exist"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/plot/suggest-clusters",
                json={"collection_name": "not_found"},
            )

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not_found" in response.json()["detail"]
