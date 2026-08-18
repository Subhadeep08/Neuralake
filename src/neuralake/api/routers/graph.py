import uuid

from fastapi import APIRouter, Depends, HTTPException

from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.api.schemas.common import PaginationParams
from neuralake.api.schemas.requests import GraphTraversalRequest
from neuralake.api.schemas.responses import GraphEntityResponse, GraphNeighborResponse
from neuralake.api.dependencies import get_graph_entity_repo
from neuralake.storage.repositories.graph import GraphEntityRepository

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/entities", response_model=list[GraphEntityResponse])
async def list_entities(
    search: str | None = None,
    entity_type: str | None = None,
    pagination: PaginationParams = Depends(),
    auth: AuthContext = Depends(get_auth_context),
    repo: GraphEntityRepository = Depends(get_graph_entity_repo),
):
    if search:
        entity = await repo.find_by_name(
            name=search, tenant_id=auth.tenant_id, entity_type=entity_type
        )
        return [GraphEntityResponse.model_validate(entity)] if entity else []

    items, total = await repo.list(
        tenant_id=auth.tenant_id,
        offset=pagination.offset,
        limit=pagination.limit,
        **({"entity_type": entity_type} if entity_type else {}),
    )
    return [GraphEntityResponse.model_validate(i) for i in items]


@router.get("/entities/{entity_id}", response_model=GraphEntityResponse)
async def get_entity(
    entity_id: uuid.UUID,
    auth: AuthContext = Depends(get_auth_context),
    repo: GraphEntityRepository = Depends(get_graph_entity_repo),
):
    entity = await repo.get(entity_id)
    if not entity or entity.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Entity not found")
    return GraphEntityResponse.model_validate(entity)


@router.get(
    "/entities/{entity_id}/neighbors",
    response_model=list[GraphNeighborResponse],
)
async def get_neighbors(
    entity_id: uuid.UUID,
    max_depth: int = 2,
    limit: int = 50,
    auth: AuthContext = Depends(get_auth_context),
    repo: GraphEntityRepository = Depends(get_graph_entity_repo),
):
    entity = await repo.get(entity_id)
    if not entity or entity.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Entity not found")

    neighbors = await repo.get_neighbors(
        entity_id=entity_id,
        tenant_id=auth.tenant_id,
        max_depth=max_depth,
        limit=limit,
    )
    return [GraphNeighborResponse(**n) for n in neighbors]


@router.post("/traverse", response_model=list[GraphNeighborResponse])
async def traverse_graph(
    body: GraphTraversalRequest,
    auth: AuthContext = Depends(get_auth_context),
    repo: GraphEntityRepository = Depends(get_graph_entity_repo),
):
    entity = await repo.get(body.entity_id)
    if not entity or entity.tenant_id != auth.tenant_id:
        raise HTTPException(status_code=404, detail="Entity not found")

    neighbors = await repo.get_neighbors(
        entity_id=body.entity_id,
        tenant_id=auth.tenant_id,
        max_depth=body.max_depth,
        relation_types=body.relation_types,
        limit=body.limit,
    )
    return [GraphNeighborResponse(**n) for n in neighbors]
