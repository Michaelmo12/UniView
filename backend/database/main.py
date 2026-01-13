import logging
from fastapi import FastAPI

from config import settings
from core.lifespan import lifespan
from core.middleware import setup_middleware
from routes import auth, users, system

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Database microservice for UniView",
    lifespan=lifespan,
)

setup_middleware(app)

app.include_router(system.router)
app.include_router(auth.router)
app.include_router(users.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.SERVICE_PORT,
        reload=True,
        log_level="info",
    )
