"""
Core modules for the API Gateway
Contains authentication and middleware components
"""
from .auth import create_jwt, verify_jwt, get_current_user, oauth2_scheme
from .middleware import setup_middleware

__all__ = [
    "create_jwt",
    "verify_jwt",
    "get_current_user",
    "oauth2_scheme",
    "setup_middleware"
]
