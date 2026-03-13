from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base_class import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="detections")

    vehicle_type = Column(String, nullable=False)
    vehicle_confidence = Column(Float, nullable=True)

    plate_text = Column(String(10), nullable=False)
    plate_confidence = Column(Float, nullable=True)
    plate_bbox = Column(JSON, nullable=True)

    original_image_s3_key = Column(String, nullable=False)
    cropped_vehicle_s3_key = Column(String, nullable=True)
    cropped_plate_s3_key = Column(String, nullable=True)

    is_allowed = Column(Boolean, default=False)
    allowed_car_id = Column(Integer, ForeignKey("allowed_cars.id"), nullable=True)
    allowed_car = relationship("AllowedCar", back_populates="detections")

    processing_time_ms = Column(Float, nullable=True)
    source = Column(String, default="upload")