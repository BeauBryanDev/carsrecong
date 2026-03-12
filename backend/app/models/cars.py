from sqlalchemy import String, Float, Boolean, DateTime, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from datetime import datetime
from app.db.base_class import Base
from app.schemas.car_schema import VehicleType  # Importing the Enum from your schema
"""
SQLAlchemy ORM model for Car detection logs.
"""
class Car(Base):
    __tablename__ = "cars"

    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    car_type: Mapped[VehicleType] = mapped_column(SQLEnum(VehicleType), nullable=False)
    color: Mapped[str | None] = mapped_column(String(30), nullable=True)
    license_plate: Mapped[str] = mapped_column(String(15), index=True, nullable=False)
    timestamp_video: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Timestamps handled at the database level
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())