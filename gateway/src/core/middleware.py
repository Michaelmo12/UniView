"""
Middleware configuration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings


def setup_middleware(app: FastAPI):
    """Configure all middleware for the application"""

    # CORS - Allow frontend to access the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.FRONTEND_URL],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
