import uuid

from fastapi import APIRouter, Depends, HTTPException

from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.api.schemas.common import PaginationParams
from neuralake.api.schemas.requests import CreateCollectionRequest, UpdateCollectionRequest
from neuralake.api.schemas.responses import CollectionListResponse, CollectionResponse
from neuralake.config.constants import EMBEDDING_DIMENSIONS
from neuralake.storage.models.collection import Collection
from neuralake.storage.repositories.base import BaseRepository
from neuralake.api.dependencies import get_collection_repo

router = APIRouter(prefix="/collections", tags=["collections"])


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(
    body: CreateCollectionRequest,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_collection_repo),
):
    dim = EMBEDDING_DIMENSIONS.get(body.embedding_model, 1536)
    collection = await repo.create(
        tenant_id=auth.tenant_id,
        name=body.name,
        description=body.description,
        embedding_model=body.embedding_model,
        embedding_dim=dim,
        chunking_strategy=body.chunking_strategy,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
        metadata=body.metadata,
    )
    return CollectionResponse.model_validate(collection)


@router.get("", response_model=CollectionListResponse)
async def list_collections(
    pagination: PaginationParams = Depends(),
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_collection_repo),
):
    items, total = await repo.list(
        tenant_id=auth.tenant_id,
        offset=pagination.offset,
        limit=pagination.limit,
    )
    return CollectionListResponse(
        items=[CollectionResponse.model_validate(i) for i in items],
        total=total,
        offset=pagination.offset,
        limit=pagination.limit,
    )


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_collection_repo),
):
    collection = await repo.get(collection_id)
    if not collection or collection.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Collection not found")
    return CollectionResponse.model_validate(collection)


@router.patch("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: uuid.UUID,
    body: UpdateCollectionRequest,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_collection_repo),
):
    existing = await repo.get(collection_id)
    if not existing or existing.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Collection not found")

    updates = body.model_dump(exclude_unset=True)
    updated = await repo.update(collection_id, **updates)
    return CollectionResponse.model_validate(updated)


@router.delete("/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    repo: BaseRepository = Depends(get_collection_repo),
):
    existing = await repo.get(collection_id)
    if not existing or existing.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Collection not found")
    await repo.delete(collection_id)
