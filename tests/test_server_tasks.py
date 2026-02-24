from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from embedding_cluster.server.tasks import (
    TaskRegistry,
    TaskState,
    TaskStatus,
)
from embedding_cluster.server.ws import WebSocketManager


class TestTaskStatus:
    def test_enum_values(self) -> None:
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTaskState:
    def test_default_values(self) -> None:
        """Test TaskState default values."""
        task = TaskState(job_id="test-id")
        assert task.job_id == "test-id"
        assert task.status == TaskStatus.PENDING
        assert task.progress == {}
        assert task.result is None
        assert task.error is None
        assert isinstance(task.cancel_event, asyncio.Event)
        assert not task.cancel_event.is_set()


class TestTaskRegistry:
    def test_create(self) -> None:
        """Test TaskRegistry.create() creates task with unique ID and PENDING status."""
        registry = TaskRegistry()
        task1 = registry.create()
        task2 = registry.create()

        assert task1.job_id != task2.job_id
        assert task1.status == TaskStatus.PENDING
        assert task2.status == TaskStatus.PENDING
        assert registry.get(task1.job_id) == task1
        assert registry.get(task2.job_id) == task2

    def test_get_existing_task(self) -> None:
        """Test TaskRegistry.get() returns task by ID."""
        registry = TaskRegistry()
        task = registry.create()

        retrieved = registry.get(task.job_id)
        assert retrieved is task
        assert retrieved is not None
        assert retrieved.job_id == task.job_id

    def test_get_missing_task(self) -> None:
        """Test TaskRegistry.get() returns None for missing task."""
        registry = TaskRegistry()
        assert registry.get("nonexistent-id") is None

    def test_cancel_running_task(self) -> None:
        """Test TaskRegistry.cancel() cancels running task."""
        registry = TaskRegistry()
        task = registry.create()
        task.status = TaskStatus.RUNNING

        result = registry.cancel(task.job_id)

        assert result is True
        assert task.status == TaskStatus.CANCELLED
        assert task.cancel_event.is_set()

    def test_cancel_pending_task(self) -> None:
        """Test TaskRegistry.cancel() returns False for pending task."""
        registry = TaskRegistry()
        task = registry.create()
        # Task is PENDING by default

        result = registry.cancel(task.job_id)

        assert result is False
        assert task.status == TaskStatus.PENDING
        assert not task.cancel_event.is_set()

    def test_cancel_completed_task(self) -> None:
        """Test TaskRegistry.cancel() returns False for completed task."""
        registry = TaskRegistry()
        task = registry.create()
        task.status = TaskStatus.COMPLETED

        result = registry.cancel(task.job_id)

        assert result is False
        assert task.status == TaskStatus.COMPLETED

    def test_cancel_nonexistent_task(self) -> None:
        """Test TaskRegistry.cancel() returns False for nonexistent task."""
        registry = TaskRegistry()
        result = registry.cancel("nonexistent-id")
        assert result is False


class TestWebSocketManager:
    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        """Test WebSocketManager.connect() accepts and stores connection."""
        manager = WebSocketManager()
        ws = MagicMock()
        ws.accept = AsyncMock()

        await manager.connect("job-1", ws)

        ws.accept.assert_awaited_once()
        assert ws in manager._connections["job-1"]

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Test WebSocketManager.disconnect() removes connection."""
        manager = WebSocketManager()
        ws = MagicMock()
        ws.accept = AsyncMock()

        await manager.connect("job-1", ws)
        assert ws in manager._connections["job-1"]

        await manager.disconnect("job-1", ws)
        assert ws not in manager._connections["job-1"]

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_clients(self) -> None:
        """Test WebSocketManager.broadcast() sends to all connected clients."""
        manager = WebSocketManager()
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()
        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()

        await manager.connect("job-1", ws1)
        await manager.connect("job-1", ws2)

        data = {"status": "running", "progress": 50}
        await manager.broadcast("job-1", data)

        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_awaited_once()

        # Verify JSON serialization
        import json

        expected_json = json.dumps(data)
        ws1.send_text.assert_awaited_with(expected_json)
        ws2.send_text.assert_awaited_with(expected_json)

    @pytest.mark.asyncio
    async def test_broadcast_no_clients(self) -> None:
        """Test WebSocketManager.broadcast() handles no clients gracefully."""
        manager = WebSocketManager()
        data = {"status": "running"}

        # Should not raise exception
        await manager.broadcast("job-1", data)

    @pytest.mark.asyncio
    async def test_broadcast_handles_send_errors(self) -> None:
        """Test WebSocketManager.broadcast() handles send errors gracefully."""
        manager = WebSocketManager()
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock(side_effect=Exception("Connection closed"))
        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()

        await manager.connect("job-1", ws1)
        await manager.connect("job-1", ws2)

        data = {"status": "running"}
        # Should not raise exception even though ws1 fails
        await manager.broadcast("job-1", data)

        ws1.send_text.assert_awaited_once()
        ws2.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_connections(self) -> None:
        manager = WebSocketManager()
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock(side_effect=Exception("Connection closed"))
        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()

        await manager.connect("job-1", ws1)
        await manager.connect("job-1", ws2)

        data = {"status": "running"}
        await manager.broadcast("job-1", data)

        assert ws1 not in manager._connections["job-1"]
        assert ws2 in manager._connections["job-1"]

    @pytest.mark.asyncio
    async def test_broadcast_different_jobs(self) -> None:
        """Test WebSocketManager.broadcast() only sends to specific job_id."""
        manager = WebSocketManager()
        ws_job1 = MagicMock()
        ws_job1.accept = AsyncMock()
        ws_job1.send_text = AsyncMock()
        ws_job2 = MagicMock()
        ws_job2.accept = AsyncMock()
        ws_job2.send_text = AsyncMock()

        await manager.connect("job-1", ws_job1)
        await manager.connect("job-2", ws_job2)

        data = {"status": "running"}
        await manager.broadcast("job-1", data)

        ws_job1.send_text.assert_awaited_once()
        ws_job2.send_text.assert_not_awaited()
