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

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Must be set before OpenVINO / OpenMP loads. These cap the thread pools that
# OpenVINO uses for inference so the 4 ENet receiver threads (cv2.imdecode)
# don't starve YOLO mid-inference, causing 5000ms detection times.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENVINO_CPU_THREADS_NUM", "4")

# Raise process priority so Windows foreground-boost (e.g. Firefox full-screen)
# does not starve YOLO/OpenVINO when the algorithm runs on a single monitor.
# ABOVE_NORMAL is safer than HIGH — HIGH can cause audio glitches / input lag
# if OpenVINO threads spike during inference.
try:
    import psutil
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    print(f"[priority] algorithm process PID {p.pid} set to {p.nice()}")
except Exception as e:
    print(f"[priority] failed to set priority: {e}")

import cv2
cv2.setNumThreads(1)

# Ensure algorithm/ directory is on path so `src.*` imports resolve.
_algorithm_dir = Path(__file__).parent
if str(_algorithm_dir) not in sys.path:
    sys.path.insert(0, str(_algorithm_dir))

import uvicorn
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

    def _on_pipeline_done(t: asyncio.Task) -> None:
        if not t.cancelled() and t.exception() is not None:
            logger.error(
                "Pipeline task crashed: %s", t.exception(), exc_info=t.exception()
            )

    task.add_done_callback(_on_pipeline_done)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drone-ids",
        type=str,
        default="",
        help="Comma-separated drone IDs to connect to, e.g. 3,4,6,7 (overrides settings default)",
    )
    args, _ = parser.parse_known_args()

    if args.drone_ids:
        ids = [int(x) for x in args.drone_ids.split(",") if x.strip()]
        settings.ingestion.drone_ids = ids
        logger.info("Drone IDs overridden via CLI: %s", ids)

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
