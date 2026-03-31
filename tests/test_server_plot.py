from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar, cast
from unittest.mock import Mock, patch

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from fastapi import FastAPI

from embedding_cluster.server.app import create_app

T = TypeVar("T")


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
def mock_compute() -> Iterator[None]:
    def fake_compute(_settings: object) -> dict[str, object]:
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
async def test_start_compute_success(app: FastAPI, mock_compute: None) -> None:
    _ = mock_compute
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test_collection"},
        )

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert "job_id" in data
    assert cast("str", data["status"]) == "pending"


@pytest.mark.asyncio
async def test_start_compute_runs_in_thread(app: FastAPI) -> None:
    def fake_compute(_settings: object) -> dict[str, object]:
        return {"points": [], "clusters": [], "total_points": 0}

    with (
        patch(
            "embedding_cluster.server.routes.plot.compute_plot_data",
            side_effect=fake_compute,
        ) as compute,
        patch("embedding_cluster.server.routes.plot.asyncio.to_thread") as to_thread,
    ):

        async def fake_to_thread(
            fn: Callable[..., T],
            *args: object,
            **kwargs: object,
        ) -> T:
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
async def test_start_compute_missing_collection(app: FastAPI, mock_compute: None) -> None:
    _ = mock_compute
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/compute",
            json={},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_get_data_not_found(app: FastAPI, mock_compute: None) -> None:
    _ = mock_compute
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/plot/data/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_get_data_pending(app: FastAPI, mock_compute: None) -> None:
    _ = mock_compute
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start_response = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test_collection"},
        )
        start_data = cast("dict[str, object]", start_response.json())
        job_id = cast("str", start_data["job_id"])

        # Immediately check status
        response = await client.get(f"/api/plot/data/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert cast("bool", data["ready"]) is False
    assert cast("str", data["status"]) in ["pending", "running", "completed"]


@pytest.mark.asyncio
async def test_get_data_completed(app: FastAPI, mock_compute: None) -> None:
    _ = mock_compute
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start_response = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test_collection"},
        )
        start_data = cast("dict[str, object]", start_response.json())
        job_id = cast("str", start_data["job_id"])

        # Wait for completion
        await asyncio.sleep(0.2)

        response = await client.get(f"/api/plot/data/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert cast("bool", data["ready"]) is True
    assert cast("str", data["status"]) == "completed"
    assert "points" in data
    assert "clusters" in data
    assert "total_points" in data
    points = cast("list[object]", data["points"])
    clusters = cast("list[object]", data["clusters"])
    assert len(points) == 2
    assert len(clusters) == 2
    assert cast("int", data["total_points"]) == 2


@pytest.mark.asyncio
async def test_get_data_failed(app: FastAPI) -> None:
    def failing_compute(_settings: object) -> dict[str, object]:
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
            start_data = cast("dict[str, object]", start_response.json())
            job_id = cast("str", start_data["job_id"])

            # Wait for computation to fail
            await asyncio.sleep(0.2)

            response = await client.get(f"/api/plot/data/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert cast("bool", data["ready"]) is False
    assert cast("str", data["status"]) == "failed"
    assert "error" in data
    assert "Computation failed" in cast("str", data["error"])


@pytest.mark.asyncio
async def test_compute_with_all_fields(app: FastAPI, mock_compute: None) -> None:
    _ = mock_compute
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
            },
        )

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert "job_id" in data
    assert cast("str", data["status"]) == "pending"


@pytest.fixture
def mock_suggest() -> Iterator[None]:
    def fake_suggest(
        _embeddings: object,
        k_range: range,
        max_samples: int = 5000,
        on_progress: object | None = None,
    ) -> dict[str, object]:
        _ = (max_samples, on_progress)
        return {
            "k_values": list(k_range),
            "inertias": [100.0 - i * 10 for i in range(len(k_range))],
            "silhouette_scores": [0.3 + i * 0.05 for i in range(len(k_range))],
            "suggested_k": max(k_range),
        }

    with patch(
        "embedding_cluster.server.routes.plot.suggest_optimal_clusters",
        side_effect=fake_suggest,
    ):
        yield


@pytest.fixture
def mock_chromadb_collection() -> Iterator[Mock]:
    import numpy as np

    mock_embeddings = cast(
        "list[list[float]]",
        np.random.default_rng(42).random((20, 10)).tolist(),
    )

    with patch(
        "embedding_cluster.server.routes.plot.load_chromadb_embeddings"
    ) as mock_load:
        mock_load.return_value = np.array(mock_embeddings)
        yield mock_load


@pytest.mark.asyncio
async def test_suggest_clusters_success(
    app: FastAPI, mock_suggest: None, mock_chromadb_collection: Mock
) -> None:
    _ = (mock_suggest, mock_chromadb_collection)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/suggest-clusters",
            json={"collection_name": "test_collection"},
        )

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert "job_id" in data
    assert cast("str", data["status"]) == "pending"


@pytest.mark.asyncio
async def test_suggest_clusters_custom_range(
    app: FastAPI, mock_suggest: None, mock_chromadb_collection: Mock
) -> None:
    _ = (mock_suggest, mock_chromadb_collection)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start_response = await client.post(
            "/api/plot/suggest-clusters",
            json={"collection_name": "test_collection", "k_min": 3, "k_max": 15},
        )
        start_data = cast("dict[str, object]", start_response.json())
        job_id = cast("str", start_data["job_id"])

        await asyncio.sleep(0.3)

        response = await client.get(f"/api/plot/suggest-clusters/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert cast("bool", data["ready"]) is True
    result = cast("dict[str, object]", data["result"])
    k_values = cast("list[int]", result["k_values"])
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
        side_effect=ValueError("Collection not found"),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            start_response = await client.post(
                "/api/plot/suggest-clusters",
                json={"collection_name": "nonexistent"},
            )
            start_data = cast("dict[str, object]", start_response.json())
            job_id = cast("str", start_data["job_id"])

            await asyncio.sleep(0.3)

            response = await client.get(f"/api/plot/suggest-clusters/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert cast("bool", data["ready"]) is False
    assert cast("str", data["status"]) == "failed"
    assert "error" in data


@pytest.mark.asyncio
async def test_suggest_clusters_status_not_found(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/plot/suggest-clusters/nonexistent")

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_suggest_clusters_status_completed(
    app: FastAPI, mock_suggest: None, mock_chromadb_collection: Mock
) -> None:
    _ = (mock_suggest, mock_chromadb_collection)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start_response = await client.post(
            "/api/plot/suggest-clusters",
            json={"collection_name": "test_collection"},
        )
        start_data = cast("dict[str, object]", start_response.json())
        job_id = cast("str", start_data["job_id"])

        await asyncio.sleep(0.3)

        response = await client.get(f"/api/plot/suggest-clusters/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert cast("bool", data["ready"]) is True
    assert cast("str", data["status"]) == "completed"
    result = cast("dict[str, object]", data["result"])
    assert "k_values" in result
    assert "suggested_k" in result


@pytest.mark.asyncio
async def test_get_data_strips_internal_fields(app: FastAPI) -> None:
    """Internal fields should not leak to the frontend response."""

    def fake_compute_with_internals(_settings: object) -> dict[str, object]:
        return {
            "points": [
                {"x": 1.0, "y": 2.0, "z": 3.0, "cluster": 0, "metadata": {}, "id": "1"},
            ],
            "clusters": [{"index": 0, "name": "A", "color": "red", "count": 1}],
            "total_points": 1,
            "embeddings_standardized": [[0.1, 0.2]],
            "cluster_labels": [0],
            "point_ids": ["1"],
        }

    with patch(
        "embedding_cluster.server.routes.plot.compute_plot_data",
        side_effect=fake_compute_with_internals,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            start_resp = await client.post(
                "/api/plot/compute",
                json={"chromadb_collection_name": "test_collection"},
            )
            job_id = cast("str", start_resp.json()["job_id"])

            await asyncio.sleep(0.2)

            response = await client.get(f"/api/plot/data/{job_id}")

    assert response.status_code == status.HTTP_200_OK
    data = cast("dict[str, object]", response.json())
    assert cast("bool", data["ready"]) is True
    # Public fields present
    assert "points" in data
    assert "clusters" in data
    assert "total_points" in data
    # Internal fields stripped
    assert "embeddings_standardized" not in data
    assert "cluster_labels" not in data
    assert "point_ids" not in data
