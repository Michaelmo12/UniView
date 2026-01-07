"""
API Routes for the Gateway
Defines all API endpoints
"""
import time
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel
from typing import Dict, Any
import httpx
from src.core.auth import create_jwt, get_current_user
from src.config import settings


router = APIRouter()


class LoginRequest(BaseModel):
    """Login request model"""
    email: str
    password: str


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class ProtectedResponse(BaseModel):
    """Protected route response model"""
    message: str
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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.BACKEND_URL}/auth/login",
                json={"email": credentials.email, "password": credentials.password},
                timeout=10.0
            )

            if response.status_code == 200:
                user = response.json()

                # Create JWT token with user data
                user_data = {
                    "user_id": user["id"],
                    "email": user["email"],
                    "role": user["role"]
                }

                token = create_jwt(user_data)

                return LoginResponse(
                    access_token=token,
                    token_type="bearer",
                    user=user
                )
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect email or password",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Backend authentication error"
                )

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Backend service unavailable: {str(e)}"
        )


@router.get("/api/protected", response_model=ProtectedResponse)
async def protected_route(current_user: Dict = Depends(get_current_user)):
    """
    Example protected route that requires JWT authentication

    Args:
        current_user: User info from JWT token (injected by dependency)

    Returns:
        Protected data accessible only to authenticated users
    """
    return ProtectedResponse(
        message="You have access to this protected resource",
        user=current_user
    )


@router.get("/health")
async def health_check(response: Response):
    """
    Dynamic health check endpoint
    Tests gateway functionality and backend connectivity
    """
    health_status = {
        "status": "healthy",
        "service": "gateway",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }

    all_healthy = True

    # Check 1: Backend Connectivity
    try:
        start_time = time.time()
        async with httpx.AsyncClient() as client:
            backend_response = await client.get(
                f"{settings.BACKEND_URL}/health",
                timeout=5.0
            )

        backend_time = round((time.time() - start_time) * 1000, 2)

        if backend_response.status_code == 200:
            health_status["checks"]["backend"] = {
                "status": "healthy",
                "response_time_ms": backend_time,
                "url": settings.BACKEND_URL
            }
        else:
            all_healthy = False
            health_status["checks"]["backend"] = {
                "status": "unhealthy",
                "status_code": backend_response.status_code,
                "url": settings.BACKEND_URL
            }
    except Exception as e:
        all_healthy = False
        health_status["checks"]["backend"] = {
            "status": "unreachable",
            "error": str(e),
            "url": settings.BACKEND_URL
        }

    # Check 2: JWT Configuration
    try:
        if settings.JWT_SECRET_KEY == "your-secret-key-change-this":
            health_status["checks"]["jwt"] = {
                "status": "warning",
                "message": "Using default JWT secret key - change in production!"
            }
        else:
            health_status["checks"]["jwt"] = {
                "status": "healthy",
                "algorithm": settings.JWT_ALGORITHM,
                "expiration_minutes": settings.JWT_EXPIRATION_MINUTES
            }
    except Exception as e:
        all_healthy = False
        health_status["checks"]["jwt"] = {
            "status": "unhealthy",
            "error": str(e)
        }

    # Overall status
    if not all_healthy:
        health_status["status"] = "unhealthy"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return health_status


@router.get("/api/backend-health")
async def backend_health_check(current_user: Dict = Depends(get_current_user)):
    """
    Check backend database service health (requires authentication)

    Args:
        current_user: User info from JWT token

    Returns:
        Backend service health status
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.BACKEND_URL}/health", timeout=5.0)
            return {
                "gateway": "healthy",
                "backend": response.json() if response.status_code == 200 else "unhealthy",
                "backend_status_code": response.status_code
            }
    except Exception as e:
        return {
            "gateway": "healthy",
            "backend": "unreachable",
            "error": str(e)
        }


@router.post("/api/users")
async def add_user(user_data: Dict[str, Any], current_user: Dict = Depends(get_current_user)):
    """
    Add a new user (Admin only - proxied to backend)

    Args:
        user_data: User signup data
        current_user: Current authenticated user (must be admin)

    Returns:
        Created user data
    """
    # TODO: Add admin role check
    # if current_user.get("role") != "admin":
    #     raise HTTPException(status_code=403, detail="Admin access required")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.BACKEND_URL}/auth/add-user",
                json=user_data,
                timeout=10.0
            )

            if response.status_code == 201:
                return response.json()
            elif response.status_code == 400:
                raise HTTPException(status_code=400, detail=response.json().get("detail", "Bad request"))
            else:
                # Try to extract detailed error from backend response
                error_detail = "Backend error"
                try:
                    backend_error = response.json()
                    if isinstance(backend_error, dict) and "detail" in backend_error:
                        error_detail = backend_error["detail"]
                    elif isinstance(backend_error, str):
                        error_detail = backend_error
                except:
                    error_detail = f"Backend error (status {response.status_code})"
                raise HTTPException(status_code=response.status_code, detail=error_detail)

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend service unavailable: {str(e)}")


@router.get("/api/users/{user_id}")
async def get_user_by_id(user_id: int, current_user: Dict = Depends(get_current_user)):
    """
    Get user by ID (proxied to backend)

    Args:
        user_id: User ID
        current_user: Current authenticated user

    Returns:
        User data
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.BACKEND_URL}/users/{user_id}",
                timeout=5.0
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
                except:
                    error_detail = f"Backend error (status {response.status_code})"
                raise HTTPException(status_code=response.status_code, detail=error_detail)

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend service unavailable: {str(e)}")


@router.get("/api/users/email/{email}")
async def get_user_by_email(email: str, current_user: Dict = Depends(get_current_user)):
    """
    Get user by email (proxied to backend)

    Args:
        email: User email
        current_user: Current authenticated user

    Returns:
        User data
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.BACKEND_URL}/users/email/{email}",
                timeout=5.0
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
                except:
                    error_detail = f"Backend error (status {response.status_code})"
                raise HTTPException(status_code=response.status_code, detail=error_detail)

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend service unavailable: {str(e)}")


@router.delete("/api/users/{user_id}")
async def delete_user(user_id: int, current_user: Dict = Depends(get_current_user)):
    """
    Delete user by ID (Admin only - proxied to backend)

    Args:
        user_id: User ID to delete
        current_user: Current authenticated user (must be admin)

    Returns:
        No content
    """
    # TODO: Add admin role check
    # if current_user.get("role") != "admin":
    #     raise HTTPException(status_code=403, detail="Admin access required")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{settings.BACKEND_URL}/users/{user_id}",
                timeout=5.0
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
                except:
                    error_detail = f"Backend error (status {response.status_code})"
                raise HTTPException(status_code=response.status_code, detail=error_detail)

    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Backend service unavailable: {str(e)}")
