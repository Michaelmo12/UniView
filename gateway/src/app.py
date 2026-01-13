from fastapi import FastAPI
from src.core import setup_middleware
from src.api import router

app = FastAPI(
    title="UniView API Gateway",
    description="API Gateway with JWT Authentication",
    version="1.0.0"
)

# Setup middleware (CORS, etc.)
setup_middleware(app)

# Include routes
app.include_router(router)

@app.get("/")
async def root():
    return {
        "service": "UniView API Gateway",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "login": "/login",
            "users": "/users",
            "docs": "/docs"
        }
    }
