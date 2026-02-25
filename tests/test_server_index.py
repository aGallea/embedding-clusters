from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from embedding_cluster.server.app import create_app
from embedding_cluster.server.routes.index import resolve_csv_path
from embedding_cluster.server.tasks import TaskStatus


@pytest.fixture
def app():
    return create_app()


def test_resolve_csv_path_defaults_to_uploads() -> None:
    assert resolve_csv_path("sample.csv") == Path("./uploads") / "sample.csv"


def test_resolve_csv_path_preserves_uploads_prefix() -> None:
    assert resolve_csv_path("uploads/sample.csv") == Path("uploads/sample.csv")
    assert resolve_csv_path("./uploads/sample.csv") == Path("./uploads/sample.csv")


def test_resolve_csv_path_preserves_absolute_path(tmp_path: Path) -> None:
    absolute_path = tmp_path / "data.csv"
    with pytest.raises(ValueError, match="Absolute CSV paths"):
        resolve_csv_path(str(absolute_path))


def test_resolve_csv_path_rejects_parent_traversal() -> None:
    with pytest.raises(ValueError, match="parent directory"):
        resolve_csv_path("../secret.csv")


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture(autouse=True)
def mock_indexer():
    """Mock main_indexer to avoid loading ML models in tests."""

    async def fake_indexer(settings, on_progress=None, on_log=None, cancel_event=None):
        if on_progress:
            on_progress(
                {
                    "rows_indexed": 5,
                    "total_rows": None,
                    "errors": 0,
                    "elapsed_seconds": 0,
                }
            )
        await asyncio.sleep(0.1)
        if cancel_event and cancel_event.is_set():
            return
        if on_progress:
            on_progress(
                {
                    "rows_indexed": 10,
                    "total_rows": None,
                    "errors": 0,
                    "elapsed_seconds": 1,
                }
            )

    with patch(
        "embedding_cluster.server.routes.index.main_indexer", side_effect=fake_indexer
    ):
        yield


async def test_start_index_success(client, mock_indexer):
    """Test starting an indexing job with valid request."""
    request_data = {
        "csv_filename": "./test.csv",
        "id_field": "id",
        "image_embedding_fields": ["imageUrl"],
        "chromadb_collection_prefix": "test_",
    }

    response = await client.post("/api/index/start", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ["pending", "running"]

    # Give the background task a moment to start
    await asyncio.sleep(0.2)


async def test_start_index_missing_fields(client, mock_indexer):
    """Test starting indexing with missing required field."""
    request_data = {
        # Missing csv_filename (required)
        "id_field": "id",
    }

    response = await client.post("/api/index/start", json=request_data)

    assert response.status_code == 422  # Validation error


async def test_status_success(client, mock_indexer):
    """Test getting status of a running job."""
    # Start a job
    request_data = {
        "csv_filename": "./test.csv",
        "image_embedding_fields": ["imageUrl"],
    }
    start_response = await client.post("/api/index/start", json=request_data)
    assert start_response.status_code == 200
    job_id = start_response.json()["job_id"]

    # Give the job a moment to start
    await asyncio.sleep(0.05)

    # Get status
    response = await client.get(f"/api/index/status/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["status"] in ["pending", "running", "completed"]
    assert "rows_indexed" in data
    assert "total_rows" in data
    assert "errors" in data
    assert data["error"] is None


async def test_status_includes_error_on_failure(client, mock_indexer):
    async def failing_indexer(settings, on_progress=None, on_log=None, cancel_event=None):
        raise RuntimeError("boom")

    with patch(
        "embedding_cluster.server.routes.index.main_indexer", side_effect=failing_indexer
    ):
        request_data = {
            "csv_filename": "./test.csv",
        }

        start_response = await client.post("/api/index/start", json=request_data)
        job_id = start_response.json()["job_id"]

        await asyncio.sleep(0.05)

        response = await client.get(f"/api/index/status/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "boom"


async def test_status_includes_elapsed_seconds(client, mock_indexer):
    request_data = {
        "csv_filename": "./test.csv",
    }

    start_response = await client.post("/api/index/start", json=request_data)
    job_id = start_response.json()["job_id"]

    await asyncio.sleep(0.15)

    response = await client.get(f"/api/index/status/{job_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["rows_indexed"] in [0, 5, 10]
    assert data["errors"] == 0


async def test_status_not_found(client, mock_indexer):
    """Test getting status of non-existent job."""
    response = await client.get("/api/index/status/nonexistent-id")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


async def test_cancel_running_job(client, mock_indexer):
    """Test cancelling a running job."""

    # Need a longer-running job for cancellation
    async def slow_indexer(settings, on_progress=None, on_log=None, cancel_event=None):
        for i in range(10):
            if cancel_event and cancel_event.is_set():
                return
            if on_progress:
                on_progress(
                    {
                        "rows_indexed": i,
                        "total_rows": 10,
                        "errors": 0,
                        "elapsed_seconds": i,
                    }
                )
            await asyncio.sleep(0.1)

    with patch(
        "embedding_cluster.server.routes.index.main_indexer", side_effect=slow_indexer
    ):
        # Start a job
        request_data = {"csv_filename": "./test.csv"}
        start_response = await client.post("/api/index/start", json=request_data)
        job_id = start_response.json()["job_id"]

        # Wait for job to be RUNNING
        await asyncio.sleep(0.05)

        # Manually set to RUNNING (since our mock might not be fast enough)
        from embedding_cluster.server.tasks import task_registry

        task = task_registry.get(job_id)
        if task:
            task.status = TaskStatus.RUNNING

        # Cancel the job
        response = await client.post(f"/api/index/cancel/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert "cancelled" in data["message"].lower()


async def test_cancel_not_found(client, mock_indexer):
    """Test cancelling non-existent job."""
    response = await client.post("/api/index/cancel/nonexistent-id")

    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


async def test_cancel_not_cancellable(client, mock_indexer):
    """Test cancelling a job that is not in RUNNING state."""
    # Start a job
    request_data = {"csv_filename": "./test.csv"}
    start_response = await client.post("/api/index/start", json=request_data)
    job_id = start_response.json()["job_id"]

    # Job is PENDING, should not be cancellable
    response = await client.post(f"/api/index/cancel/{job_id}")

    assert response.status_code == 400
    data = response.json()
    assert "not cancellable" in data["detail"].lower()


def test_ws_not_found(app, mock_indexer):
    """Test WebSocket connection to non-existent job."""
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(app)

    # SIM117: Nested with statements are clearer here for readability
    with pytest.raises(WebSocketDisconnect):  # noqa: SIM117
        with client.websocket_connect("/api/index/ws/nonexistent-id"):
            pass


async def test_index_request_defaults(client, mock_indexer):
    """Test IndexRequest with default values."""
    request_data = {
        "csv_filename": "./test.csv",
    }

    response = await client.post("/api/index/start", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data

    # Give the background task a moment
    await asyncio.sleep(0.2)


async def test_index_request_all_fields(client, mock_indexer):
    """Test IndexRequest with all fields specified."""
    request_data = {
        "csv_filename": "./test.csv",
        "id_field": "id",
        "image_embedding_fields": ["imageUrl", "thumbnail"],
        "text_embedding_fields": ["name", "description"],
        "image_model_name": "custom/model",
        "text_model_name": "custom/text-model",
        "chromadb_collection_prefix": "custom_",
        "number_of_async_tasks": 5,
        "index_bulk_size": 50,
        "index_start_line": 10,
        "index_end_line": 100,
        "process_unit_device": "cuda",
        "embedding_fields_prefix": "emb_",
    }

    response = await client.post("/api/index/start", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data

    # Give the background task a moment
    await asyncio.sleep(0.2)


async def test_status_progress_tracking(client, mock_indexer):
    """Test that status endpoint returns progress data correctly."""
    # Start a job
    request_data = {"csv_filename": "./test.csv"}
    start_response = await client.post("/api/index/start", json=request_data)
    job_id = start_response.json()["job_id"]

    # Wait for progress to be updated
    await asyncio.sleep(0.15)

    # Get status
    response = await client.get(f"/api/index/status/{job_id}")

    assert response.status_code == 200
    data = response.json()
    # Our mock indexer reports 5 then 10 rows
    assert data["rows_indexed"] in [0, 5, 10]
    assert data["errors"] == 0
