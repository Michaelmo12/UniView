"""
Pydantic schemas for API request/response validation
These are NOT database models - they validate data coming from/to the API
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

# Request Schemas (data coming FROM the client)

class UserSignup(BaseModel):
    """
    Schema for user signup request
    Used when creating a new user account
    """
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")
    full_name: str = Field(..., min_length=2, description="Full name")
    role: Optional[str] = Field(default="user", description="User role: user or admin")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securePassword123",
                "full_name": "John Doe",
                "role": "user"
            }
        }

class UserLogin(BaseModel):
    """
    Schema for user login request
    """
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "securePassword123"
            }
        }

# Response Schemas (data going TO the client)

class UserResponse(BaseModel):
    """
    Schema for user data in responses
    Never send password_hash to the client!
    """
    id: int
    email: str
    full_name: str
    role: str
    created_at: datetime

    class Config:
        from_attributes = True  # Allows conversion from SQLAlchemy models

class Token(BaseModel):
    """
    Schema for JWT token response
    """
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    """
    Schema for data stored inside JWT token
    """
    email: Optional[str] = None
    user_id: Optional[int] = None

class LoginResponse(BaseModel):
    """
    Schema for login response
    Returns both user data and access token
    """
    user: UserResponse
    access_token: str
    token_type: str = "bearer"
