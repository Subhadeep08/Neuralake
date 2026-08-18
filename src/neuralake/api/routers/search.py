from fastapi import APIRouter, Depends

from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.api.schemas.requests import QueryRequest, SearchRequest
from neuralake.api.schemas.responses import QueryResponse, SearchResult
from neuralake.api.dependencies import get_vector_repo
from neuralake.storage.repositories.vector import VectorRepository

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResult)
async def hybrid_search(
    body: SearchRequest,
    auth: AuthContext = Depends(get_auth_context),
    vector_repo: VectorRepository = Depends(get_vector_repo),
):
    from neuralake.core.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(vector_repo=vector_repo, db=vector_repo.session)
    result = await pipeline.search(
        query=body.query,
        tenant_id=auth.tenant_id,
        collection_ids=body.collection_ids,
        top_k=body.top_k,
        min_score=body.min_score,
        include_memories=body.include_memories,
        include_graph=body.include_graph,
    )
    return result


@router.post("/query", response_model=QueryResponse)
async def rag_query(
    body: QueryRequest,
    auth: AuthContext = Depends(get_auth_context),
    vector_repo: VectorRepository = Depends(get_vector_repo),
):
    from neuralake.core.query.engine import QueryEngine

    engine = QueryEngine(vector_repo=vector_repo, db=vector_repo.session)
    result = await engine.query(
        query=body.query,
        tenant_id=auth.tenant_id,
        user_id=body.user_id or auth.user_id,
        collection_ids=body.collection_ids,
        top_k=body.top_k,
        include_citations=body.include_citations,
        include_memories=body.include_memories,
        include_graph=body.include_graph,
        conversation_history=body.conversation_history,
    )
    return result
