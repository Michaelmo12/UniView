from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from core import get_db, hash_password, authenticate_user, create_access_token
from models import User
from schemas import UserSignup, UserLogin, UserResponse, LoginResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """
    Register a new user account

    - Validates email format and password length
    - Checks if email already exists
    - Hashes password before storing
    - Creates user in database
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

    logger.info(f"New user registered: {new_user.email}")

    return new_user

@router.post("/login", response_model=LoginResponse)
def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Login with email and password

    - Authenticates user credentials
    - Returns user data and JWT access token
    - Token valid for 24 hours (configurable in .env)
    """
    user = authenticate_user(db, credentials.email, credentials.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": user.email, "user_id": user.id}
    )

    logger.info(f"User logged in: {user.email}")

    return {
        "user": user,
        "access_token": access_token,
        "token_type": "bearer"
    }
