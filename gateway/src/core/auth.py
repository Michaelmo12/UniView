# Handle token expiration times
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import logging

# library for jwt tokens
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends

# Extracts Bearer token from Authorization header
from fastapi.security import OAuth2PasswordBearer
from src.config import settings

# is_token_blacklisted function to check if token is blacklisted
from .token_blacklist import is_token_blacklisted, TokenBlacklistError

# Configure logging
logger = logging.getLogger(__name__)

# Tells FastAPI where to get tokens. This extracts the token from Authorization: Bearer <token> header.
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
        "exp": datetime.now(timezone.utc)
        + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES),  # expiration time
        "iat": datetime.now(timezone.utc),  # issued at,
    }
    # Creates JWT token
    token = jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
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
        HTTPException: If token is invalid, expired, blacklisted, or blacklist check fails
    """

    # Check if token is blacklisted (logged out)
    try:
        if is_token_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except TokenBlacklistError as e:
        # Database error checking blacklist - fail closed for security
        logger.error(f"Token blacklist check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )

        user_id: int = payload.get("user_id")
        email: str = payload.get("email")
        role: str = payload.get("role")
        # later make special exceptions for different missing fields
        if user_id is None or email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return {"user_id": user_id, "email": email, "role": role}
    
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


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


async def get_admin_user(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated admin user
    Use this with Depends() on admin-only routes

    This dependency chains with get_current_user, so it:
    1. First validates the JWT token (via get_current_user)
    2. Then checks if the user has admin role

    Args:
        current_user: User info from get_current_user dependency

    Returns:
        User information dictionary (guaranteed to be admin)

    Raises:
        HTTPException: If user is not an admin (403 Forbidden)

    Example:
        @router.post("/api/users")
        async def add_user(admin: Dict = Depends(get_admin_user)):
            # This code only runs if user is admin
            pass
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user
