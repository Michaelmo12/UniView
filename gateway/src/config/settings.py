from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    BaseSettings automatically loads from .env file and environment variables.
    """

    # JWT Settings
    JWT_SECRET_KEY: str = "your-secret-key-change-this"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60

    # Backend Service URLs
    BACKEND_URL: str = "http://localhost:8000"

    # CORS Settings
    FRONTEND_URL: str = "http://localhost:5173"

    # Gateway Settings
    GATEWAY_HOST: str = "0.0.0.0"
    GATEWAY_PORT: int = 8080

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
