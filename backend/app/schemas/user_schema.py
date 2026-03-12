from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    """Base schema containing common attributes for a User."""
    full_name: str = Field(..., min_length=2, max_length=100)
    username: str = Field(..., min_length=2, max_length=100)
    email: EmailStr = Field(..., min_length=5, max_length=100)
    phone_number: Optional[str] = Field(None, max_length=20)
    gender: Optional[str] = Field(None, pattern="male|female")
    country: Optional[str] = Field(None, max_length=255)
    address: Optional[str] = Field(None, max_length=255)
    is_active: bool = True

class UserCreate(UserBase):
    """Schema for creating a new User. Includes sensitive data."""
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    """Schema for updating an existing User."""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    username: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = Field(None, min_length=5, max_length=100)
    password: Optional[str] = Field(None, min_length=8)
    phone_number: Optional[str] = Field(None, max_length=20)
    gender: Optional[str] = Field(None, pattern="male|female")
    country: Optional[str] = Field(None, max_length=255)
    address: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    """Schema for returning User data. Excludes sensitive data like passwords."""
    id: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)