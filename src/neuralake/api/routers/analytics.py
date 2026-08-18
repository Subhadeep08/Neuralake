from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.api.schemas.responses import UsageStatsResponse
from neuralake.storage.database import get_db
from neuralake.storage.models.analytics import UsageEvent
from neuralake.storage.models.document import Chunk, Document
from neuralake.storage.models.graph import GraphEntity
from neuralake.storage.models.memory import Memory

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/usage", response_model=UsageStatsResponse)
async def get_usage_stats(
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db),
):
    doc_count = (
        await db.execute(
            select(func.count()).where(Document.tenant_id == auth.tenant_id)
        )
    ).scalar() or 0

    chunk_count = (
        await db.execute(
            select(func.count()).where(Chunk.tenant_id == auth.tenant_id)
        )
    ).scalar() or 0

    memory_count = (
        await db.execute(
            select(func.count()).where(Memory.tenant_id == auth.tenant_id)
        )
    ).scalar() or 0

    entity_count = (
        await db.execute(
            select(func.count()).where(GraphEntity.tenant_id == auth.tenant_id)
        )
    ).scalar() or 0

    events_result = await db.execute(
        select(UsageEvent)
        .where(UsageEvent.tenant_id == auth.tenant_id)
        .order_by(UsageEvent.created_at.desc())
        .limit(50)
    )
    events = [
        {"event_type": e.event_type, "details": e.details, "created_at": str(e.created_at)}
        for e in events_result.scalars().all()
    ]

    return UsageStatsResponse(
        total_documents=doc_count,
        total_chunks=chunk_count,
        total_memories=memory_count,
        total_entities=entity_count,
        events=events,
    )
