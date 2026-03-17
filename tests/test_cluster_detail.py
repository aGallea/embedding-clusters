from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from collections.abc import Iterator

    from fastapi import FastAPI

from embedding_cluster.server.app import create_app


@pytest.fixture
def app() -> FastAPI:
    return create_app()


def _fake_compute_with_internals(
    _settings: object,
) -> dict[str, object]:
    """Fake compute that includes internal fields."""
    # 4 points, 2 clusters
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.1],
        ]
    )
    return {
        "points": [
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "cluster": 0,
                "metadata": {"name": "item1"},
                "id": "1",
            },
            {
                "x": 0.1,
                "y": 0.1,
                "z": 0.1,
                "cluster": 0,
                "metadata": {"name": "item2"},
                "id": "2",
            },
            {
                "x": 1.0,
                "y": 1.0,
                "z": 1.0,
                "cluster": 1,
                "metadata": {"name": "item3"},
                "id": "3",
            },
            {
                "x": 1.1,
                "y": 1.1,
                "z": 1.1,
                "cluster": 1,
                "metadata": {"name": "item4"},
                "id": "4",
            },
        ],
        "clusters": [
            {
                "index": 0,
                "name": "Group 1",
                "color": "hsl(0, 70%, 50%)",
                "count": 2,
            },
            {
                "index": 1,
                "name": "Group 2",
                "color": "hsl(180, 70%, 50%)",
                "count": 2,
            },
        ],
        "total_points": 4,
        "embeddings_standardized": embeddings.tolist(),
        "cluster_labels": [0, 0, 1, 1],
        "point_ids": ["1", "2", "3", "4"],
    }


@pytest.fixture
def mock_compute_internals() -> Iterator[None]:
    with patch(
        "embedding_cluster.server.routes.plot.compute_plot_data",
        side_effect=_fake_compute_with_internals,
    ):
        yield


@pytest.mark.asyncio
async def test_cluster_detail_success(app: FastAPI, mock_compute_internals: None) -> None:
    _ = mock_compute_internals
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test"},
        )
        job_id = cast("str", start.json()["job_id"])
        await asyncio.sleep(0.2)

        response = await client.get(f"/api/plot/{job_id}/cluster/0")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["cluster_index"] == 0
    assert data["cluster_name"] == "Group 1"
    assert data["total_items"] == 2
    assert len(data["items"]) == 2
    # Items should be sorted by distance to centroid
    distances = [item["distance_to_centroid"] for item in data["items"]]
    assert distances == sorted(distances)


@pytest.mark.asyncio
async def test_cluster_detail_pagination(
    app: FastAPI, mock_compute_internals: None
) -> None:
    _ = mock_compute_internals
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test"},
        )
        job_id = cast("str", start.json()["job_id"])
        await asyncio.sleep(0.2)

        response = await client.get(f"/api/plot/{job_id}/cluster/0?page=1&page_size=1")

    data = response.json()
    assert data["page"] == 1
    assert data["page_size"] == 1
    assert len(data["items"]) == 1
    assert data["total_items"] == 2


@pytest.mark.asyncio
async def test_cluster_detail_invalid_cluster(
    app: FastAPI, mock_compute_internals: None
) -> None:
    _ = mock_compute_internals
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test"},
        )
        job_id = cast("str", start.json()["job_id"])
        await asyncio.sleep(0.2)

        response = await client.get(f"/api/plot/{job_id}/cluster/99")

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_cluster_detail_invalid_job(
    app: FastAPI,
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/plot/nonexistent/cluster/0")

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_cluster_detail_job_not_ready(
    app: FastAPI, mock_compute_internals: None
) -> None:
    _ = mock_compute_internals
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test"},
        )
        job_id = cast("str", start.json()["job_id"])
        # Don't wait - query immediately
        response = await client.get(f"/api/plot/{job_id}/cluster/0")

    # Should get 409 or 200 depending on timing
    assert response.status_code in (
        status.HTTP_409_CONFLICT,
        status.HTTP_200_OK,
    )


def _fake_compute_interleaved_clusters(
    _settings: object,
) -> dict[str, object]:
    """Fake compute with interleaved cluster labels.

    Regression fixture: simulates real KMeans output where labels are NOT
    grouped (e.g. [0, 1, 0, 1, 0, 1]).  The old code built points grouped
    by cluster, so points[i] and cluster_labels[i] referred to different
    items — causing the cluster-detail endpoint to return wrong products.
    """
    n = 6
    rng = np.random.default_rng(99)
    embeddings = rng.random((n, 4))
    labels = [0, 1, 0, 1, 0, 1]

    points = []
    for i in range(n):
        points.append(
            {
                "x": float(i),
                "y": float(i),
                "z": float(i),
                "cluster": labels[i],
                "metadata": {"name": f"item{i}"},
                "id": str(i),
            }
        )

    return {
        "points": points,
        "clusters": [
            {
                "index": 0,
                "name": "Group 1",
                "color": "hsl(0, 70%, 50%)",
                "count": 3,
            },
            {
                "index": 1,
                "name": "Group 2",
                "color": "hsl(180, 70%, 50%)",
                "count": 3,
            },
        ],
        "total_points": n,
        "embeddings_standardized": embeddings.tolist(),
        "cluster_labels": labels,
        "point_ids": [str(i) for i in range(n)],
    }


@pytest.fixture
def mock_compute_interleaved() -> Iterator[None]:
    with patch(
        "embedding_cluster.server.routes.plot.compute_plot_data",
        side_effect=_fake_compute_interleaved_clusters,
    ):
        yield


@pytest.mark.asyncio
async def test_cluster_detail_returns_correct_items_with_interleaved_labels(
    app: FastAPI, mock_compute_interleaved: None
) -> None:
    """Regression: cluster-detail must return items belonging to the
    requested cluster when cluster_labels are interleaved (not grouped).
    """
    _ = mock_compute_interleaved
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test"},
        )
        job_id = cast("str", start.json()["job_id"])
        await asyncio.sleep(0.2)

        resp_c0 = await client.get(f"/api/plot/{job_id}/cluster/0")
        resp_c1 = await client.get(f"/api/plot/{job_id}/cluster/1")

    assert resp_c0.status_code == status.HTTP_200_OK
    assert resp_c1.status_code == status.HTTP_200_OK

    c0_data = resp_c0.json()
    c1_data = resp_c1.json()

    c0_ids = {item["id"] for item in c0_data["items"]}
    c1_ids = {item["id"] for item in c1_data["items"]}

    assert c0_ids == {"0", "2", "4"}, (
        f"Cluster 0 should contain items 0,2,4 but got {c0_ids}"
    )
    assert c1_ids == {"1", "3", "5"}, (
        f"Cluster 1 should contain items 1,3,5 but got {c1_ids}"
    )
