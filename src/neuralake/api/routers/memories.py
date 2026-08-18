import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.api.schemas.common import PaginationParams
from neuralake.api.schemas.requests import (
    CreateMemoryRequest,
    ExtractMemoriesRequest,
    SearchMemoriesRequest,
    UpdateMemoryRequest,
)
from neuralake.api.schemas.responses import (
    MemoryListResponse,
    MemoryResponse,
    MemorySearchResult,
)
from neuralake.api.dependencies import get_memory_repo, get_vector_repo
from neuralake.storage.repositories.base import BaseRepository
from neuralake.storage.repositories.vector import VectorRepository

router = APIRouter(prefix="/memories", tags=["memories"])


@router.post("", response_model=MemoryResponse, status_code=201)
async def create_memory(
    body: CreateMemoryRequest,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_memory_repo),
):
    from neuralake.core.embeddings.registry import get_embedder

    embedder = get_embedder()
    embedding = await embedder.embed_text(body.content)

    memory = await repo.create(
        tenant_id=auth.tenant_id,
        user_id=auth.user_id,
        memory_type=body.memory_type,
        content=body.content,
        category=body.category,
        importance=body.importance,
        confidence=1.0,
        source=body.source or "api",
        source_ref=body.source_ref,
        embedding=embedding,
        metadata=body.metadata,
        valid_from=body.valid_from or datetime.now(timezone.utc),
        steps=body.steps,
        trigger_conditions=body.trigger_conditions,
    )
    return MemoryResponse.model_validate(memory)


@router.get("", response_model=MemoryListResponse)
async def list_memories(
    memory_type: str | None = None,
    pagination: PaginationParams = Depends(),
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_memory_repo),
):
    filters = {}
    if memory_type:
        filters["memory_type"] = memory_type
    items, total = await repo.list(
        tenant_id=auth.tenant_id,
        offset=pagination.offset,
        limit=pagination.limit,
        **filters,
    )
    return MemoryListResponse(
        items=[MemoryResponse.model_validate(i) for i in items],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.post("/search", response_model=list[MemorySearchResult])
async def search_memories(
    body: SearchMemoriesRequest,
    auth: AuthContext = Depends(get_auth_context),
    vector_repo: VectorRepository = Depends(get_vector_repo),
):
    from neuralake.core.embeddings.registry import get_embedder

    embedder = get_embedder()
    embedding = await embedder.embed_text(body.query)

    results = await vector_repo.search_memories(
        embedding=embedding,
        tenant_id=auth.tenant_id,
        user_id=body.user_id,
        memory_types=body.memory_types,
        top_k=body.top_k,
        min_score=body.min_score,
    )
    return [
        MemorySearchResult(
            memory=MemoryResponse.model_validate(mem),
            score=score,
        )
        for mem, score in results
    ]


@router.post("/extract", response_model=list[MemoryResponse])
async def extract_memories(
    body: ExtractMemoriesRequest,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_memory_repo),
):
    from neuralake.core.memory.extractor import MemoryExtractor

    extractor = MemoryExtractor()
    extracted = await extractor.extract(
        messages=body.messages,
        tenant_id=auth.tenant_id,
        user_id=body.user_id or auth.user_id,
        extract_types=body.extract_types,
        db=repo.session,
    )
    return [MemoryResponse.model_validate(m) for m in extracted]


@router.post("/consolidate", response_model=dict)
async def consolidate_memories(
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_memory_repo),
):
    from neuralake.core.memory.consolidator import MemoryConsolidator

    consolidator = MemoryConsolidator()
    stats = await consolidator.consolidate(
        tenant_id=auth.tenant_id,
        db=repo.session,
    )
    return stats


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_memory_repo),
):
    memory = await repo.get(memory_id)
    if not memory or memory.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryResponse.model_validate(memory)


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: uuid.UUID,
    body: UpdateMemoryRequest,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_memory_repo),
):
    existing = await repo.get(memory_id)
    if not existing or existing.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Memory not found")

    updates = body.model_dump(exclude_unset=True)
    if "content" in updates:
        from neuralake.core.embeddings.registry import get_embedder

        embedder = get_embedder()
        updates["embedding"] = await embedder.embed_text(updates["content"])

    updated = await repo.update(memory_id, **updates)
    return MemoryResponse.model_validate(updated)


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_memory_repo),
):
    existing = await repo.get(memory_id)
    if not existing or existing.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Memory not found")
    await repo.delete(memory_id)
