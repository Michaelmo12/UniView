"""
WebSocket Relay

Two responsibilities:
1. upstream_relay_task: Background coroutine that connects to the algorithm WebSocket
   (ws://localhost:8001/ws) and relays messages to all connected frontend clients.
   Uses websockets auto-reconnect pattern (async for ws in connect(...)).

2. ws_stream endpoint: FastAPI WebSocket endpoint at /ws/stream that:
   - Validates JWT from query param before accepting
   - Registers client with ws_manager
   - Keeps connection alive until client disconnects

JWT is in query param (not Authorization header) because WebSocket browser clients
cannot set custom headers — this is FastAPI's documented pattern for WS auth.
"""
import logging

import websockets.asyncio.client
import websockets.exceptions
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from jose import JWTError, jwt

from src.config import settings
from src.core.ws_manager import ws_manager

logger = logging.getLogger(__name__)

ws_router = APIRouter()


async def upstream_relay_task() -> None:
    """
    Connect to algorithm WebSocket and relay messages to frontend clients.

    The 'async for ws in connect(...)' pattern from websockets 15.x provides
    automatic reconnection with backoff — do NOT use a manual retry loop.
    """
    logger.info("Upstream relay task starting. Connecting to %s", settings.ALGORITHM_WS_URL)

    async for ws in websockets.asyncio.client.connect(settings.ALGORITHM_WS_URL):
        try:
            logger.info("Connected to algorithm WebSocket at %s", settings.ALGORITHM_WS_URL)
            async for message in ws:
                if ws_manager.num_connections > 0:
                    await ws_manager.broadcast_text(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Algorithm WebSocket connection closed. Reconnecting...")
            continue


@ws_router.websocket("/ws/stream")
async def ws_stream(
    websocket: WebSocket,
    token: str = Query(..., description="JWT access token"),
) -> None:
    """
    Authenticated WebSocket endpoint for frontend clients.

    Validates JWT before accepting. Clients receive the same messages
    that the algorithm broadcasts — frames + tracking data.

    Usage (browser): new WebSocket('ws://localhost:8080/ws/stream?token=<jwt>')
    """
    # Validate JWT before accepting connection
    try:
        jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
    except JWTError:
        await websocket.close(code=1008)   # 1008 = Policy Violation
        logger.warning("Rejected WebSocket connection: invalid JWT")
        return

    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep alive; ignore client messages
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
