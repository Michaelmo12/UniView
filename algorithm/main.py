"""
Algorithm Microservice Entry Point

FastAPI application:
- Runs the full pipeline (ingestion -> detection -> features -> fusion -> reconstruction -> tracking)
  as a background asyncio task.
- After each frame, POSTs one StreamPayload per drone to the gateway POST /api/internal/push.
- GET /health: liveness probe.

Run with (from UniView/ root):
    uvicorn algorithm.main:app --host 0.0.0.0 --port 8001
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure algorithm/ directory is on path so `src.*` imports resolve.
_algorithm_dir = Path(__file__).parent
if str(_algorithm_dir) not in sys.path:
    sys.path.insert(0, str(_algorithm_dir))

from fastapi import FastAPI

from src.api.gateway_client import close_client, init_client
from src.config.settings import settings
from src.pipeline.inference_pipeline import run_pipeline_loop

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize gateway client, start pipeline task; clean up on shutdown."""
    logger.info("Initializing gateway HTTP client: %s", settings.output.gateway_url)
    await init_client(base_url=settings.output.gateway_url)

    logger.info("Starting algorithm pipeline background task...")
    task = asyncio.create_task(run_pipeline_loop())
    yield

    logger.info("Shutting down pipeline...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    await close_client()
    logger.info("Pipeline shutdown complete.")


app = FastAPI(
    title="UniView Algorithm Microservice",
    description="Multi-drone person tracking — POSTs StreamPayload to gateway per frame",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Liveness probe."""
    return {
        "status": "pass",
        "service": "algorithm",
        "version": "1.0.0",
    }
