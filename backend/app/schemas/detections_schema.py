from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any


class DetectionBase(BaseModel):
    vehicle_type: str
    vehicle_confidence: Optional[float] = None
    plate_text: str = Field(..., min_length=5, max_length=10)
    plate_confidence: Optional[float] = None
    original_image_s3_key: str
    cropped_vehicle_s3_key: Optional[str] = None
    cropped_plate_s3_key: Optional[str] = None
    is_allowed: bool = False
    processing_time_ms: Optional[float] = None
    source: str = "upload"


class DetectionCreate(DetectionBase):

    user_id: Optional[int] = None
    allowed_car_id: Optional[int] = None
    plate_bbox: Optional[Dict[str, Any]] = None


class Detection(DetectionBase):

    id: int
    created_at: datetime
    user_id: Optional[int] = None
    allowed_car_id: Optional[int] = None
    plate_bbox: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True