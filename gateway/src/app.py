from fastapi import FastAPI
from src.core import setup_middleware
from src.api import router


app = FastAPI(
    title="UniView API Gateway",
    description="API Gateway with JWT Authentication",
    version="1.0.0",
)

setup_middleware(app)
app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "UniView API Gateway",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "login": "/api/login",
            "users": "/api/users",
            "docs": "/docs",
            "push": "/api/internal/push",
            "stream": "/stream/live",
        }
    }
