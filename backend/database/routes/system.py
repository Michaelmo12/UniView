"""
System routes (health check, root)
"""
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException, status, Response
from sqlalchemy import text
from config import settings
from core import engine

router = APIRouter(tags=["System"])


@router.get("/health")
def health_check(response: Response):
    """
    Dynamic health check endpoint
    Tests actual database connectivity and returns detailed status
    """
    health_status = {
        "status": "healthy",
        "service": "database",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }

    all_healthy = True

    # Check 1: Database Connection
    try:
        start_time = time.time()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        db_response_time = round((time.time() - start_time) * 1000, 2)  # ms

        health_status["checks"]["database"] = {
            "status": "healthy",
            "response_time_ms": db_response_time,
            "connection_pool": {
                "size": engine.pool.size(),
                "checked_in": engine.pool.checkedin(),
                "overflow": engine.pool.overflow()
            }
        }
    except Exception as e:
        all_healthy = False
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }

    # Check 2: Database Tables
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public'"
            ))
            tables = [row[0] for row in result]

        health_status["checks"]["tables"] = {
            "status": "healthy",
            "count": len(tables),
            "tables": tables
        }
    except Exception as e:
        health_status["checks"]["tables"] = {
            "status": "warning",
            "error": str(e)
        }

    # Overall status
    if not all_healthy:
        health_status["status"] = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return health_status


@router.get("/")
def root():
    """Root endpoint - basic info"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "users": "/users",
            "auth": "/auth"
        }
    }
