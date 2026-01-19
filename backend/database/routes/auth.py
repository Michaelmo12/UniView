from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from core import get_db, hash_password, authenticate_user
from models import User
from schemas import UserSignup, UserLogin, UserResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/add-user", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def add_user(user_data: UserSignup, db: Session = Depends(get_db)):
    """
    Add a new user (Admin only)

    - Validates email format and password length
    - Checks if email already exists
    - Hashes password before storing
    - Creates user in database

    Note: This endpoint should only be accessible to admin users.
    Access control is handled by the Gateway.
    """
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    hashed_password = hash_password(user_data.password)

    new_user = User(
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info(f"New user added by admin: {new_user.email}")

    return new_user

@router.post("/login", response_model=UserResponse)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Validate user credentials

    Called by Gateway to validate login.
    Gateway creates JWT token, not this service.
    Returns user data if credentials are valid.
    """
    user = authenticate_user(db, credentials.email, credentials.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    logger.info(f"User credentials validated: {user.email}")

    return user
