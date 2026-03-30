from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from httpx import ASGITransport, AsyncClient

if TYPE_CHECKING:
    from fastapi import FastAPI

from embedding_cluster.server.app import create_app
from embedding_cluster.server.tasks import TaskStatus, task_registry


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
def completed_job() -> str:
    """Create a completed job in the task registry and return its ID."""
    task = task_registry.create()
    task.status = TaskStatus.COMPLETED
    task.result = {
        "points": [
            {
                "id": "p1",
                "x": 1.0,
                "y": 2.0,
                "z": 3.0,
                "cluster": 0,
                "metadata": {"name": "Running Shoes"},
            },
            {
                "id": "p2",
                "x": 4.0,
                "y": 5.0,
                "z": 6.0,
                "cluster": 0,
                "metadata": {"name": "Basketball Sneakers"},
            },
            {
                "id": "p3",
                "x": 7.0,
                "y": 8.0,
                "z": 9.0,
                "cluster": 1,
                "metadata": {"name": "Summer Dress"},
            },
        ],
        "cluster_labels": [0, 0, 1],
        "clusters": [
            {"index": 0, "name": "Group 1", "color": "#ff0000", "count": 2},
            {"index": 1, "name": "Group 2", "color": "#00ff00", "count": 1},
        ],
        "total_points": 3,
    }
    return task.job_id


def _mock_llm_response(content: str = "Athletic Footwear") -> MagicMock:
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response.choices = [mock_choice]
    return mock_response


class TestNameClusters:
    @pytest.mark.asyncio
    async def test_names_clusters_successfully(
        self, app: FastAPI, completed_job: str
    ) -> None:
        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=_mock_llm_response("Athletic Footwear"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/ai/name-clusters",
                    json={
                        "job_id": completed_job,
                        "cluster_indices": [0],
                        "api_key": "test-key",
                        "model": "gpt-4o-mini",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = cast("dict[str, object]", response.json())
        names = cast("dict[str, str]", data["names"])
        assert "0" in names
        assert names["0"] == "Athletic Footwear"

    @pytest.mark.asyncio
    async def test_names_multiple_clusters(
        self, app: FastAPI, completed_job: str
    ) -> None:
        call_count = 0

        def side_effect(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_llm_response("Athletic Footwear")
            return _mock_llm_response("Fashion Dresses")

        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            side_effect=side_effect,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/ai/name-clusters",
                    json={
                        "job_id": completed_job,
                        "cluster_indices": [0, 1],
                        "api_key": "test-key",
                        "model": "gpt-4o-mini",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = cast("dict[str, object]", response.json())
        names = cast("dict[str, str]", data["names"])
        assert "0" in names
        assert "1" in names

    @pytest.mark.asyncio
    async def test_job_not_found(self, app: FastAPI) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ai/name-clusters",
                json={
                    "job_id": "nonexistent-id",
                    "cluster_indices": [0],
                    "api_key": "test-key",
                    "model": "gpt-4o-mini",
                },
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_job_not_completed(self, app: FastAPI) -> None:
        task = task_registry.create()
        task.status = TaskStatus.RUNNING

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ai/name-clusters",
                json={
                    "job_id": task.job_id,
                    "cluster_indices": [0],
                    "api_key": "test-key",
                    "model": "gpt-4o-mini",
                },
            )

        assert response.status_code == status.HTTP_409_CONFLICT

    @pytest.mark.asyncio
    async def test_passes_optional_params(self, app: FastAPI, completed_job: str) -> None:
        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=_mock_llm_response("Name"),
        ) as mock_completion:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(
                    "/api/ai/name-clusters",
                    json={
                        "job_id": completed_job,
                        "cluster_indices": [0],
                        "api_key": "test-key",
                        "model": "gpt-4o-mini",
                        "base_url": "http://localhost:11434",
                        "temperature": 0.8,
                    },
                )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"
        assert call_kwargs["temperature"] == 0.8


class TestNameSubClusters:
    @pytest.mark.asyncio
    async def test_names_sub_clusters_successfully(
        self, app: FastAPI, completed_job: str
    ) -> None:
        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=_mock_llm_response("Running Shoes"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/ai/name-sub-clusters",
                    json={
                        "job_id": completed_job,
                        "point_ids": ["p1", "p2"],
                        "sub_cluster_labels": [0, 1],
                        "api_key": "test-key",
                        "model": "gpt-4o-mini",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = cast("dict[str, object]", response.json())
        names = cast("dict[str, str]", data["names"])
        assert "0" in names
        assert "1" in names

    @pytest.mark.asyncio
    async def test_with_parent_cluster_name(
        self, app: FastAPI, completed_job: str
    ) -> None:
        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=_mock_llm_response("Running Shoes"),
        ) as mock_completion:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(
                    "/api/ai/name-sub-clusters",
                    json={
                        "job_id": completed_job,
                        "point_ids": ["p1"],
                        "sub_cluster_labels": [0],
                        "api_key": "test-key",
                        "model": "gpt-4o-mini",
                        "parent_cluster_name": "Athletic Footwear",
                    },
                )

        call_kwargs = mock_completion.call_args[1]
        system_msg = call_kwargs["messages"][0]["content"]
        assert "Athletic Footwear" in system_msg

    @pytest.mark.asyncio
    async def test_job_not_found(self, app: FastAPI) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ai/name-sub-clusters",
                json={
                    "job_id": "nonexistent-id",
                    "point_ids": ["p1"],
                    "sub_cluster_labels": [0],
                    "api_key": "test-key",
                    "model": "gpt-4o-mini",
                },
            )

        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestTestConnection:
    @pytest.mark.asyncio
    async def test_successful_connection(self, app: FastAPI) -> None:
        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=_mock_llm_response("Hello"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/ai/test-connection",
                    json={
                        "api_key": "test-key",
                        "model": "gpt-4o-mini",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = cast("dict[str, object]", response.json())
        assert data["success"] is True
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_failed_connection(self, app: FastAPI) -> None:
        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            side_effect=Exception("Connection refused"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/ai/test-connection",
                    json={
                        "api_key": "test-key",
                        "model": "gpt-4o-mini",
                    },
                )

        assert response.status_code == status.HTTP_200_OK
        data = cast("dict[str, object]", response.json())
        assert data["success"] is False
        assert data["error"] is not None
        assert "Connection refused" in cast("str", data["error"])

    @pytest.mark.asyncio
    async def test_with_base_url(self, app: FastAPI) -> None:
        with patch(
            "embedding_cluster.ai_naming.litellm_completion",
            return_value=_mock_llm_response("Hello"),
        ) as mock_completion:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(
                    "/api/ai/test-connection",
                    json={
                        "api_key": "test-key",
                        "model": "gpt-4o-mini",
                        "base_url": "http://localhost:11434",
                    },
                )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["api_base"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, app: FastAPI) -> None:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/ai/test-connection",
                json={},
            )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
