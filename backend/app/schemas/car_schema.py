"""
Pydantic schemas for Car detection logs from the video stream.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from enum import Enum
from datetime import datetime

class VehicleType(str, Enum):
    """Enumeration for supported vehicle classifications."""
    CAR = "CAR"
    TRUCK = "TRUCK"
    BUS = "BUS"
    MOTORCYCLE = "MOTORCYCLE"
    

class CarBase(BaseModel):
    """Base schema for a vehicle detection event."""
    car_type: VehicleType
    color: Optional[str] = Field(None, max_length=30)
    license_plate: str = Field(..., max_length=15)
    timestamp_video: float = Field(..., description="Second in the video where detection occurred")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
    is_allowed: bool = False

class CarCreate(CarBase):
    """Schema for registering a new detection from the ML pipeline."""
    pass

class CarResponse(CarBase):
    """Schema for returning detection data to the frontend."""
    id: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)