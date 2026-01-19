"""
Database configuration using SQLAlchemy ORM
This file sets up the connection to PostgreSQL
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import settings

# Database URL from settings
DATABASE_URL = settings.DATABASE_URL

# Interface to the database.
# echo=True prints sql queries to the console for debugging
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=15, echo=True)

# workspace for database operations. transaction manager.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all database models
# All your models will inherit from this
Base = declarative_base()

# for dependency injection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """
    Create all tables in the database
    Runs once when setting up the database
    """
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")
