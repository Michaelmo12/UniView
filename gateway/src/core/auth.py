"""
JWT Authentication Module
Handles JWT token creation and verification for the API Gateway
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from src.config import settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")


def create_jwt(user_data: Dict[str, Any]) -> str:
    """
    Create a JWT token for authenticated user

    Args:
        user_data: Dictionary containing user information (user_id, email, role)

    Returns:
        JWT token string
    """
    payload = {
        "user_id": user_data.get("user_id"),
        "email": user_data.get("email"),
        "role": user_data.get("role"),
        "exp": datetime.utcnow() + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES),
        "iat": datetime.utcnow()
    }

    token = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )

    return token


def verify_jwt(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        user_id: int = payload.get("user_id")
        email: str = payload.get("email")
        role: str = payload.get("role")

        if user_id is None or email is None:
            raise credentials_exception

        return {
            "user_id": user_id,
            "email": email,
            "role": role
        }

    except JWTError:
        raise credentials_exception


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user
    Use this with Depends() on protected routes

    Args:
        token: JWT token from Authorization header

    Returns:
        User information dictionary

    Raises:
        HTTPException: If authentication fails
    """
    return verify_jwt(token)
