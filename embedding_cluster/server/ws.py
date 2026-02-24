from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = defaultdict(list)

    async def connect(self, job_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[job_id].append(websocket)

    async def disconnect(self, job_id: str, websocket: WebSocket) -> None:
        self._connections[job_id].remove(websocket)

    async def broadcast(self, job_id: str, data: dict[str, Any]) -> None:
        connections = list(self._connections.get(job_id, []))
        if not connections:
            return

        failed: list[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                failed.append(ws)
                logger.warning(
                    "Failed to send WebSocket message for job %s",
                    job_id,
                )

        if failed:
            remaining = [
                ws for ws in self._connections.get(job_id, []) if ws not in failed
            ]
            if remaining:
                self._connections[job_id] = remaining
            else:
                self._connections.pop(job_id, None)


ws_manager = WebSocketManager()
