"""
UniView Algorithm Server

FastAPI application that:
- Runs the full inference pipeline as an asyncio background task (lifespan)
- Streams tracking results over WebSocket at /ws
- Exposes /health for liveness checks

Start with:
    uvicorn algorithm.main:app --port 8001
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from algorithm.api.websocket import manager
from algorithm.pipeline.inference_pipeline import run_pipeline_loop
from algorithm.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    On startup: creates the pipeline background task.
    On shutdown: cancels the task and waits for clean exit.
    """
    logger.info(
        "Starting UniView algorithm server (port=%d)", settings.output.websocket_port
    )
    pipeline_task = asyncio.create_task(run_pipeline_loop(manager))
    try:
        yield
    finally:
        logger.info("Shutting down pipeline loop")
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass
        logger.info("Server shutdown complete")


app = FastAPI(
    title="UniView Algorithm Server",
    description="WebSocket server streaming cross-camera tracking results",
    version="1.0.0",
    lifespan=lifespan,
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming tracking results.

    Clients connect here and receive JSON messages at ~2 FPS containing
    raw drone frames (as base64 JPEG data URLs) and tracked person data.
    """
    await manager.connect(websocket)
    try:
        # Keep the connection alive until the client disconnects
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
        manager.disconnect(websocket)


@app.get("/health")
async def health() -> dict:
    """Liveness check endpoint."""
    return {
        "status": "ok",
        "connected_clients": manager.num_connections,
        "websocket_port": settings.output.websocket_port,
    }
