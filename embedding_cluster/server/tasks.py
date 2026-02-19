from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskState:
    job_id: str
    status: TaskStatus = TaskStatus.PENDING
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskState] = {}

    def create(self) -> TaskState:
        job_id = str(uuid.uuid4())
        task = TaskState(job_id=job_id)
        self._tasks[job_id] = task
        return task

    def get(self, job_id: str) -> TaskState | None:
        return self._tasks.get(job_id)

    def cancel(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        if task and task.status == TaskStatus.RUNNING:
            task.cancel_event.set()
            task.status = TaskStatus.CANCELLED
            return True
        return False


task_registry = TaskRegistry()
