from sqlalchemy import Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from neuralake.storage.models.base import TenantBase


class UsageEvent(TenantBase):
    __tablename__ = "usage_events"

    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    details: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_usage_tenant_time", "tenant_id", "created_at"),
    )
