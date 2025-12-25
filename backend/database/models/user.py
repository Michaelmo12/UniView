"""
Database models using SQLAlchemy ORM
Models define the structure of database tables as Python classes
"""
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from core.database import Base

class User(Base):
    """
    User model - represents the 'users' table in PostgreSQL

    Each attribute becomes a column in the database
    """
    __tablename__ = "users"  # Table name in PostgreSQL

    # Columns
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default='user')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        """String representation of User object (for debugging)"""
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"
