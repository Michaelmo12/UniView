"""
Application settings
Centralized configuration from environment variables
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_NAME: str = os.getenv("DB_NAME", "uniview_db")
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "admin")

    # Service
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", 8000))

    # CORS - Allow gateway to access
    CORS_ORIGINS: list = ["http://localhost:8080", "http://localhost:5173", "http://localhost:3000"]

    # Application
    APP_NAME: str = "UniView Database Microservice"
    APP_VERSION: str = "1.0.0"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

settings = Settings()
