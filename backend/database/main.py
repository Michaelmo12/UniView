"""
UniView Database Microservice
Main FastAPI application entry point
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from config import settings
from core import engine, Base
from routes import auth, users

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown of the application"""
    # STARTUP
    logger.info("=" * 60)
    logger.info("STARTING UNIVIEW DATABASE MICROSERVICE")
    logger.info("=" * 60)

    try:
        # Check if RSA keys exist, generate if not
        private_key_path = settings.JWT_PRIVATE_KEY_PATH
        public_key_path = settings.JWT_PUBLIC_KEY_PATH

        if not os.path.exists(private_key_path) or not os.path.exists(public_key_path):
            logger.info("RSA keys not found, generating new keys...")
            os.makedirs(os.path.dirname(private_key_path), exist_ok=True)

            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization

            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )

            # Write private key
            with open(private_key_path, 'wb') as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))

            # Write public key
            public_key = private_key.public_key()
            with open(public_key_path, 'wb') as f:
                f.write(public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))

            logger.info("RSA keys generated successfully")
        else:
            logger.info("RSA keys found")

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

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Database microservice for UniView drone surveillance system",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(users.router)

# Health check endpoint
@app.get("/health", tags=["System"])
def health_check():
    """
    Health check endpoint
    - Used by Docker: HEALTHCHECK
    - Used by Kubernetes: liveness/readiness probes
    - Used by load balancers: to check if service is alive
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "service": "database",
            "database": "connected"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {str(e)}"
        )

# Root endpoint
@app.get("/", tags=["System"])
def root():
    """Root endpoint - basic info"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.SERVICE_PORT,
        reload=False,
        log_level="info"
    )
