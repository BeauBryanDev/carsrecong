
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
"""
Database session management and connection pooling for PostgreSQL.
"""

# Retrieve database connection parameters from environment variables
# In a production environment, this is often handled by a centralized config module
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "secretpassword")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_DB = os.getenv("POSTGRES_DB", "carsrecong_db")

# Construct the PostgreSQL connection URI
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"

# Initialize the SQLAlchemy engine
# pool_size: Number of connections to keep open permanently
# max_overflow: Number of extra connections to create if the pool is exhausted
# pool_pre_ping: Verifies connection health before using it, which helps to avoid using stale connections
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """
    Dependency generator to yield a database session for FastAPI endpoints.
    
    Yields:
        Session: An active SQLAlchemy database session.
        
    Ensures that the session is properly closed and returned to the pool
    after the HTTP request is completed, regardless of exceptions.
    """
    db = SessionLocal()
    
    try:
        
        yield db
        
    finally:
        
        db.close()