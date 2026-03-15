# tests/test_server_annotations.py
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi import FastAPI

from embedding_cluster.server.app import create_app


@pytest.fixture
def app(tmp_path: Path) -> Iterator[FastAPI]:
    with patch(
        "embedding_cluster.server.routes.annotations._DEFAULT_ANNOTATIONS_DIR",
        tmp_path / "annotations",
    ):
        yield create_app()


@pytest.mark.asyncio
async def test_get_annotations_empty(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/annotations/somejob")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["job_id"] == "somejob"
    assert data["clusters"] == {}


@pytest.mark.asyncio
async def test_update_annotation(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put(
            "/api/annotations/somejob/cluster/0",
            json={"name": "Shoes", "notes": "Athletic shoes"},
        )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["clusters"]["0"]["name"] == "Shoes"
    assert data["clusters"]["0"]["notes"] == "Athletic shoes"


@pytest.mark.asyncio
async def test_update_partial_annotation(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.put(
            "/api/annotations/somejob/cluster/0",
            json={"name": "Shoes"},
        )
        response = await client.put(
            "/api/annotations/somejob/cluster/0",
            json={"notes": "Running shoes"},
        )

    data = response.json()
    assert data["clusters"]["0"]["name"] == "Shoes"
    assert data["clusters"]["0"]["notes"] == "Running shoes"


@pytest.mark.asyncio
async def test_delete_annotations(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.put(
            "/api/annotations/somejob/cluster/0",
            json={"name": "Shoes"},
        )
        response = await client.delete("/api/annotations/somejob")

    assert response.status_code == status.HTTP_200_OK

    # Verify empty after delete
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/annotations/somejob")

    data = response.json()
    assert data["clusters"] == {}


@pytest.mark.asyncio
async def test_annotation_with_tags(app: FastAPI) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put(
            "/api/annotations/somejob/cluster/0",
            json={"tags": ["footwear", "sport"]},
        )

    data = response.json()
    assert data["clusters"]["0"]["tags"] == ["footwear", "sport"]
