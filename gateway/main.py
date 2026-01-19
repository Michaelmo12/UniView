import uvicorn
from src.config import settings

if __name__ == "__main__":
    print("🚀 Starting UniView API Gateway...")
    print(f"📍 Server: http://localhost:{settings.GATEWAY_PORT}")
    print(f"📚 API Docs: http://localhost:{settings.GATEWAY_PORT}/docs")
    print("🛑 Press CTRL+C to stop\n")

    uvicorn.run(
        "src.app:app",
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        reload=True
    )
