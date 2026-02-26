"""Dataset model for user-uploaded CSVs and stored analytics."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.sqlite import JSON

from api.db import Base


def _uuid_str():
    return str(uuid.uuid4())


class DatasetStatus:
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)  # original CSV filename
    file_path = Column(String(512), nullable=False)  # path on disk
    status = Column(String(32), nullable=False, default=DatasetStatus.IN_PROGRESS)
    error_message = Column(Text, nullable=True)  # set when status=failed
    analytics_json = Column(JSON, nullable=True)  # dashboard payload when status=completed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "filename": self.filename,
            "filePath": self.file_path,
            "status": self.status,
            "errorMessage": self.error_message,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
        }
