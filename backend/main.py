import logging
from fastapi import FastAPI

from src.config import settings
from src.core.lifespan import lifespan
from src.core.middleware import setup_middleware
from src.routes import auth, users, system, history

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
app.include_router(history.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.SERVICE_PORT,
        reload=True,
        log_level="info",
    )
