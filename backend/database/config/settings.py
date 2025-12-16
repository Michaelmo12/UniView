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
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", 8001))

    # JWT - Asymmetric RSA
    JWT_PRIVATE_KEY_PATH: str = os.getenv("JWT_PRIVATE_KEY_PATH", "keys/private.pem")
    JWT_PUBLIC_KEY_PATH: str = os.getenv("JWT_PUBLIC_KEY_PATH", "keys/public.pem")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "RS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 1440))

    # CORS
    CORS_ORIGINS: list = ["http://localhost:5173", "http://localhost:3000"]

    # Application
    APP_NAME: str = "UniView Database Microservice"
    APP_VERSION: str = "1.0.0"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

settings = Settings()
