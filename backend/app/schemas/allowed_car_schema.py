"""
Pydantic schemas for the Auto Gate white-list simulation.
"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional

class AllowedCarBase(BaseModel):
    """Base schema for an allowed vehicle in the system."""
    license_plate: str = Field(..., min_length=3, max_length=15)

class AllowedCarCreate(AllowedCarBase):
    """Schema for adding a new license plate to the white-list."""
    owner_id: int = Field(..., description="User ID of the owner of the vehicle")

class AllowedCarResponse(AllowedCarBase):
    """Schema for returning allowed vehicle records."""
    id: int
    owner_id: int = Field(..., description="User ID of the owner of the vehicle")
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)