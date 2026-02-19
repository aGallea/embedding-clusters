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
        for ws in self._connections.get(job_id, []):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                logger.warning(
                    "Failed to send WebSocket message for job %s",
                    job_id,
                )


ws_manager = WebSocketManager()
