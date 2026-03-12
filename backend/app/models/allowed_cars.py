from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from datetime import datetime
from app.db.base_class import Base

class AllowedCar(Base):
    """
    SQLAlchemy ORM model for the Auto Gate white-list.
    """
    __tablename__ = "allowed_cars"

    id: Mapped[int] = mapped_column(primary_key=True, index=True, autoincrement=True)
    license_plate: Mapped[str] = mapped_column(String(15), unique=True, index=True, nullable=False)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship back to the user
    owner: Mapped["User"] = relationship("User", back_populates="allowed_cars")
    
    def __repr__(self):
        return f"<AllowedCar(id={self.id}, license_plate={self.license_plate}, owner_id={self.owner_id})>"