"""
Authentication and security utilities
Handles password hashing only (JWT removed - handled by Gateway)
"""
from typing import Optional, TYPE_CHECKING
from passlib.context import CryptContext
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from models import User

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Password Hashing Functions

def hash_password(password: str) -> str:
    """Hash a plain password"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(db: Session, email: str, password: str) -> Optional["User"]:
    """
    Authenticate user with email and password
    Returns User object if valid, None otherwise
    """
    from models import User
    user = db.query(User).filter(User.email == email).first()

    if not user:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user
