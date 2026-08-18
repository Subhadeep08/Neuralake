import uuid

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from neuralake.storage.models.base import TenantBase, Base


class Memory(TenantBase):
    __tablename__ = "memories"

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    memory_type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(100))
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    source: Mapped[str | None] = mapped_column(String(100))
    source_ref: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    embedding = mapped_column(Vector(1536))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    valid_from: Mapped[str | None] = mapped_column(DateTime(timezone=True))
    valid_until: Mapped[str | None] = mapped_column(DateTime(timezone=True))

    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[str | None] = mapped_column(DateTime(timezone=True))
    decay_rate: Mapped[float] = mapped_column(Float, default=0.01)
    consolidated: Mapped[bool] = mapped_column(Boolean, default=False)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id")
    )

    steps: Mapped[dict | None] = mapped_column(JSONB)
    trigger_conditions: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        Index("idx_memories_tenant_user", "tenant_id", "user_id"),
        Index("idx_memories_type", "tenant_id", "memory_type"),
        Index("idx_memories_category", "tenant_id", "category"),
        Index("idx_memories_temporal", "tenant_id", "valid_from", "valid_until"),
    )


class MemoryDerivation(Base):
    __tablename__ = "memory_derivations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    derived_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
