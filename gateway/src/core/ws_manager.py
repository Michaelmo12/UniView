"""
Frontend WebSocket Connection Manager

Manages active WebSocket connections from authenticated frontend clients.
Receives pre-serialized text messages from the upstream relay and forwards them.
"""
import logging
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active authenticated frontend WebSocket connections."""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("Frontend client connected. Total: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info("Frontend client disconnected. Total: %d", len(self.active_connections))

    async def broadcast_text(self, text: str) -> None:
        """Forward a text message to all connected frontend clients."""
        dead: list[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active_connections:
                self.active_connections.remove(ws)

    @property
    def num_connections(self) -> int:
        return len(self.active_connections)


# Module-level singleton shared between ws_relay.py and app.py
ws_manager = ConnectionManager()
