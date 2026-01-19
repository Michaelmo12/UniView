"""
Application lifespan management
Handles startup and shutdown events
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from config import settings
from core.database import engine, Base

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle events"""

    # STARTUP
    logger.info("=" * 60)
    logger.info("STARTING UNIVIEW DATABASE MICROSERVICE")
    logger.info("=" * 60)

    try:
        logger.info("Testing database connection...")
        engine.connect()
        logger.info("Database connection successful")

        logger.info("Creating database tables (if needed)...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables ready")

        logger.info("=" * 60)
        logger.info(f"SERVICE READY - Listening on port {settings.SERVICE_PORT}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise RuntimeError("Database initialization failed") from e

    yield

    # SHUTDOWN
    logger.info("Shutting down database microservice...")
    engine.dispose()
    logger.info("Database connections closed")
