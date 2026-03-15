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


def _fake_compute_for_subcluster(
    _settings: object,
) -> dict[str, object]:
    """Fake compute with enough points per cluster."""
    rng = np.random.default_rng(42)
    n_per_cluster = 20
    emb_dim = 10
    cluster0_emb = rng.normal(0, 1, (n_per_cluster, emb_dim))
    cluster1_emb = rng.normal(5, 1, (n_per_cluster, emb_dim))
    all_emb = np.vstack([cluster0_emb, cluster1_emb])

    points = []
    labels = []
    ids = []
    for i in range(n_per_cluster * 2):
        cluster = 0 if i < n_per_cluster else 1
        points.append(
            {
                "x": float(i),
                "y": float(i),
                "z": float(i),
                "cluster": cluster,
                "metadata": {"name": f"item{i}"},
                "id": str(i),
            }
        )
        labels.append(cluster)
        ids.append(str(i))

    return {
        "points": points,
        "clusters": [
            {
                "index": 0,
                "name": "Group 1",
                "color": "hsl(0, 70%, 50%)",
                "count": n_per_cluster,
            },
            {
                "index": 1,
                "name": "Group 2",
                "color": "hsl(180, 70%, 50%)",
                "count": n_per_cluster,
            },
        ],
        "total_points": n_per_cluster * 2,
        "embeddings_standardized": all_emb.tolist(),
        "cluster_labels": labels,
        "point_ids": ids,
    }


@pytest.fixture
def mock_compute_subcluster() -> Iterator[None]:
    with patch(
        "embedding_cluster.server.routes.plot.compute_plot_data",
        side_effect=_fake_compute_for_subcluster,
    ):
        yield


@pytest.mark.asyncio
async def test_sub_cluster_success(app: FastAPI, mock_compute_subcluster: None) -> None:
    _ = mock_compute_subcluster
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test"},
        )
        job_id = cast("str", start.json()["job_id"])
        await asyncio.sleep(0.2)

        response = await client.post(
            f"/api/plot/{job_id}/cluster/0/sub-cluster",
            json={"num_sub_clusters": 3},
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["parent_cluster_index"] == 0
    assert data["total_points"] == 20
    assert len(data["sub_clusters"]) == 3
    assert len(data["points"]) == 20
    # Each point has sub_cluster assignment
    for point in data["points"]:
        assert "sub_cluster" in point
        assert 0 <= point["sub_cluster"] < 3


@pytest.mark.asyncio
async def test_sub_cluster_invalid_cluster(
    app: FastAPI, mock_compute_subcluster: None
) -> None:
    _ = mock_compute_subcluster
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test"},
        )
        job_id = cast("str", start.json()["job_id"])
        await asyncio.sleep(0.2)

        response = await client.post(
            f"/api/plot/{job_id}/cluster/99/sub-cluster",
            json={"num_sub_clusters": 3},
        )

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_sub_cluster_too_few_points(
    app: FastAPI, mock_compute_subcluster: None
) -> None:
    """When num_sub_clusters > num_items, return 400."""
    _ = mock_compute_subcluster
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post(
            "/api/plot/compute",
            json={"chromadb_collection_name": "test"},
        )
        job_id = cast("str", start.json()["job_id"])
        await asyncio.sleep(0.2)

        response = await client.post(
            f"/api/plot/{job_id}/cluster/0/sub-cluster",
            json={"num_sub_clusters": 100},
        )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_sub_cluster_invalid_job(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/nonexistent/cluster/0/sub-cluster",
            json={"num_sub_clusters": 3},
        )

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_sub_cluster_validation_min(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/plot/somejob/cluster/0/sub-cluster",
            json={"num_sub_clusters": 1},
        )

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
