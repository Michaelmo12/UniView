import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Response

# Pydantic models for request/response validation
from pydantic import BaseModel
from typing import Dict, Any

# HTTP client to make requests to backend
import httpx
from src.core.auth import create_jwt, get_current_user, get_admin_user, oauth2_scheme
from src.core import add_token_to_blacklist
from src.config import settings
from jose import jwt as jose_jwt


router = APIRouter()


# todo move this basemodels to a separate file models.py
class LoginRequest(BaseModel):
    """Login request model"""

    email: str
    password: str


class LoginResponse(BaseModel):
    """Login response model"""

    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


@router.post("/api/login", response_model=LoginResponse)
async def login(credentials: LoginRequest):
    """
    Authenticate user and generate JWT token

    Forwards credentials to backend database microservice for validation,
    then generates JWT token if valid.
    """
    try:
        # Forward credentials to backend for validation
        # closes the client after use with async with
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.BACKEND_URL}/auth/login",
                json={"email": credentials.email, "password": credentials.password},
                timeout=10.0,
            )

            if response.status_code == 200:
                user = response.json()

                # Create JWT token with user data
                user_data = {
                    "user_id": user["id"],
                    "email": user["email"],
                    "role": user["role"],
                }

                token = create_jwt(user_data)

                return LoginResponse(access_token=token, token_type="bearer", user=user)
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Backend authentication error",
                )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Backend service unavailable: {str(e)}",
        )


# must run get_current_user first to extract token from header
@router.post("/api/logout")
async def logout(
    current_user: Dict = Depends(get_current_user), token: str = Depends(oauth2_scheme)
):
    """
    Logout user by blacklisting their JWT token

    Args:
        current_user: Current authenticated user
        token: JWT token to blacklist

    Returns:
        Success message
    """
    try:
        # Decode token to get expiration time
        payload = jose_jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )

        # Get expiration datetime for SQLite
        exp_timestamp = payload.get("exp")
        if exp_timestamp:
            expires_at = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)

            # Only blacklist if token hasn't expired yet
            if expires_at > datetime.now(timezone.utc):
                add_token_to_blacklist(token, expires_at)

        return {"message": "Successfully logged out", "user": current_user["email"]}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}",
        )


@router.get("/health")
async def health():
    return {
        "status": "pass",
        "service": "gateway",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/ready")
async def readiness_check(response: Response):
    """
    Readiness probe - checks if gateway can handle traffic

    Tests critical dependencies like backend connectivity.
    Returns 503 if dependencies are unhealthy.
    Used by load balancers to determine if traffic should be routed here.
    """
    health_status = {
        "status": "pass",
        "service": "gateway",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }

    all_healthy = True

    # Check 1: Backend Connectivity
    try:
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            backend_response = await client.get(
                f"{settings.BACKEND_URL}/health", timeout=5.0
            )

        backend_time = round((time.time() - start_time) * 1000, 2)

        if backend_response.status_code == 200:
            health_status["checks"]["backend"] = {
                "status": "pass",
                "response_time_ms": backend_time,
                "url": settings.BACKEND_URL,
            }
        else:
            all_healthy = False
            health_status["checks"]["backend"] = {
                "status": "fail",
                "status_code": backend_response.status_code,
                "url": settings.BACKEND_URL,
            }
    except Exception as e:
        all_healthy = False
        health_status["checks"]["backend"] = {
            "status": "fail",
            "error": str(e),
            "url": settings.BACKEND_URL,
        }

    # Overall status
    if not all_healthy:
        health_status["status"] = "fail"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return health_status


@router.post("/api/users")
async def add_user(
    user_data: Dict[str, Any], admin_user: Dict = Depends(get_admin_user)
):
    """
    Add a new user (Admin only - proxied to backend)

    Args:
        user_data: User signup data
        admin_user: Current authenticated admin user

    Returns:
        Created user data

    Raises:
        HTTPException: 403 if user is not an admin
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.BACKEND_URL}/auth/add-user", json=user_data, timeout=10.0
            )
            # returns user data if created
            if response.status_code == 201:
                return response.json()
            elif response.status_code == 400:
                # Forward validation error from backend
                raise HTTPException(
                    status_code=400, detail=response.json().get("detail", "Bad request")
                )
            else:
                # Try to extract detailed error from backend response
                error_detail = "Backend error"
                try:
                    backend_error = response.json()
                    if isinstance(backend_error, dict) and "detail" in backend_error:
                        error_detail = backend_error["detail"]
                    elif isinstance(backend_error, str):
                        error_detail = backend_error
                except (ValueError, KeyError, TypeError):
                    # If JSON parsing fails or structure is unexpected
                    error_detail = f"Backend error (status {response.status_code})"
                raise HTTPException(
                    status_code=response.status_code, detail=error_detail
                )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, detail=f"Backend service unavailable: {str(e)}"
        )


@router.get("/api/users/{user_id}")
async def get_user_by_id(user_id: int, current_user: Dict = Depends(get_current_user)):
    """
    Get user by ID (proxied to backend)

    Access control: Users can only view their own profile unless they are admin

    Args:
        user_id: User ID
        current_user: Current authenticated user

    Returns:
        User data

    Raises:
        HTTPException: 403 if user tries to view another user's profile (non-admin)
    """
    # Check if user is viewing their own profile OR is admin
    if current_user["user_id"] != user_id and current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own profile",
        )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.BACKEND_URL}/users/{user_id}", timeout=5.0
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="User not found")
            else:
                # Try to extract detailed error from backend response
                error_detail = "Backend error"
                try:
                    backend_error = response.json()
                    if isinstance(backend_error, dict) and "detail" in backend_error:
                        error_detail = backend_error["detail"]
                    elif isinstance(backend_error, str):
                        error_detail = backend_error
                except (ValueError, KeyError, TypeError):
                    # If JSON parsing fails or structure is unexpected
                    error_detail = f"Backend error (status {response.status_code})"
                raise HTTPException(
                    status_code=response.status_code, detail=error_detail
                )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, detail=f"Backend service unavailable: {str(e)}"
        )


@router.get("/api/users/email/{email}")
async def get_user_by_email(email: str, admin_user: Dict = Depends(get_admin_user)):
    """
    Get user by email (Admin only - proxied to backend)

    Args:
        email: User email
        admin_user: Current authenticated admin user

    Returns:
        User data

    Raises:
        HTTPException: 403 if user is not an admin
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.BACKEND_URL}/users/email/{email}", timeout=5.0
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="User not found")
            else:
                # Try to extract detailed error from backend response
                error_detail = "Backend error"
                try:
                    backend_error = response.json()
                    if isinstance(backend_error, dict) and "detail" in backend_error:
                        error_detail = backend_error["detail"]
                    elif isinstance(backend_error, str):
                        error_detail = backend_error
                except (ValueError, KeyError, TypeError):
                    # If JSON parsing fails or structure is unexpected
                    error_detail = f"Backend error (status {response.status_code})"
                raise HTTPException(
                    status_code=response.status_code, detail=error_detail
                )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, detail=f"Backend service unavailable: {str(e)}"
        )


@router.delete("/api/users/{user_id}")
async def delete_user(user_id: int, admin_user: Dict = Depends(get_admin_user)):
    """
    Delete user by ID (Admin only - proxied to backend)

    Args:
        user_id: User ID to delete
        admin_user: Current authenticated admin user

    Returns:
        No content

    Raises:
        HTTPException: 403 if user is not an admin
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{settings.BACKEND_URL}/users/{user_id}", timeout=5.0
            )

            if response.status_code == 204:
                return Response(status_code=204)
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="User not found")
            else:
                # Try to extract detailed error from backend response
                error_detail = "Backend error"
                try:
                    backend_error = response.json()
                    if isinstance(backend_error, dict) and "detail" in backend_error:
                        error_detail = backend_error["detail"]
                    elif isinstance(backend_error, str):
                        error_detail = backend_error
                except (ValueError, KeyError, TypeError):
                    # If JSON parsing fails or structure is unexpected
                    error_detail = f"Backend error (status {response.status_code})"
                raise HTTPException(
                    status_code=response.status_code, detail=error_detail
                )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, detail=f"Backend service unavailable: {str(e)}"
        )
