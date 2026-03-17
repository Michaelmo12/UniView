"""
WebSocket Connection Manager

Manages active WebSocket client connections and broadcasts serialized
JSON messages to all connected clients.
"""

import logging
from typing import List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages a set of active WebSocket connections.

    Thread safety: designed for use within a single asyncio event loop.
    connect/disconnect/broadcast_text are all coroutines and must be
    awaited from async context.
    """

    def __init__(self) -> None:
        self._active: List[WebSocket] = []

    @property
    def num_connections(self) -> int:
        """Number of currently connected WebSocket clients."""
        return len(self._active)

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        self._active.append(websocket)
        logger.info(
            "WebSocket client connected (total=%d)", self.num_connections
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket from the active list."""
        if websocket in self._active:
            self._active.remove(websocket)
        logger.info(
            "WebSocket client disconnected (total=%d)", self.num_connections
        )

    async def broadcast_text(self, message: str) -> None:
        """
        Send a pre-serialized string to all connected clients.

        Clients that fail to receive are disconnected silently.

        Args:
            message: JSON string to broadcast
        """
        if not self._active:
            return

        dead: List[WebSocket] = []
        for ws in list(self._active):
            try:
                await ws.send_text(message)
            except Exception as exc:
                logger.warning("Failed to send to client, dropping: %s", exc)
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)


# Module-level singleton — imported by both the endpoint and pipeline loop
manager = ConnectionManager()
