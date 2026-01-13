from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import logging

from core import get_db
from models import User
from schemas import UserResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/email/{email}", response_model=UserResponse)
def get_user_by_email(
    email: str,
    db: Session = Depends(get_db)
):
    """
    Get user by email
    No authentication - Gateway handles this
    """
    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user

@router.get("/{user_id}", response_model=UserResponse)
def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Get user by ID
    No authentication - Gateway handles this
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete user by ID (Admin only)

    Note: This endpoint should only be accessible to admin users.
    Access control is handled by the Gateway.
    """
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    db.delete(user)
    db.commit()

    logger.info(f"User deleted by admin: {user.email} (ID: {user_id})")

    return None
