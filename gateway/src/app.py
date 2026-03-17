import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from src.core import setup_middleware
from src.api import router
from src.api.ws_relay import ws_router, upstream_relay_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(upstream_relay_task())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="UniView API Gateway",
    description="API Gateway with JWT Authentication",
    version="1.0.0",
    lifespan=lifespan,
)

# Setup middleware (CORS, etc.)
setup_middleware(app)

# Include HTTP routes
app.include_router(router)

# Include WebSocket relay routes
app.include_router(ws_router)


@app.get("/")
async def root():
    return {
        "service": "UniView API Gateway",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "login": "/login",
            "users": "/users",
            "docs": "/docs",
            "stream": "/ws/stream",
        }
    }
