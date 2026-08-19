import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.api.schemas.responses import (
    ChunkResponse,
    MemoryResponse,
    MemorySearchResult,
    SearchResult,
)
from neuralake.core.embeddings.registry import get_embedder
from neuralake.core.retrieval.fusion import reciprocal_rank_fusion
from neuralake.storage.repositories.vector import VectorRepository


class RetrievalPipeline:
    def __init__(self, vector_repo: VectorRepository, db: AsyncSession):
        self.vector_repo = vector_repo
        self.db = db
        self.embedder = get_embedder()

    async def search(
        self,
        query: str,
        tenant_id: uuid.UUID,
        collection_ids: list[uuid.UUID] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
        include_memories: bool = True,
        include_graph: bool = True,
    ) -> SearchResult:
        embedding = await self.embedder.embed_text(query)

        vector_results = await self.vector_repo.search_chunks(
            embedding=embedding,
            tenant_id=tenant_id,
            collection_ids=collection_ids,
            top_k=top_k * 2,
            min_score=min_score,
        )

        sparse_results = await self.vector_repo.sparse_search_chunks(
            query_text=query,
            tenant_id=tenant_id,
            collection_ids=collection_ids,
            top_k=top_k * 2,
        )

        vector_pairs = [(str(c.id), s) for c, s in vector_results]
        sparse_pairs = [(str(c.id), s) for c, s in sparse_results]

        fused = reciprocal_rank_fusion(vector_pairs, sparse_pairs)

        chunk_map = {}
        for chunk, score in vector_results:
            chunk_map[str(chunk.id)] = (chunk, score)
        for chunk, score in sparse_results:
            if str(chunk.id) not in chunk_map:
                chunk_map[str(chunk.id)] = (chunk, score)

        chunks = []
        for chunk_id, rrf_score in fused[:top_k]:
            if chunk_id in chunk_map:
                chunk, orig_score = chunk_map[chunk_id]
                chunks.append(
                    ChunkResponse(
                        id=chunk.id,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        token_count=chunk.token_count,
                        score=rrf_score,
                        metadata=chunk.metadata_,
                    )
                )

        memory_results = None
        if include_memories:
            from neuralake.core.memory.engine import MemoryEngine

            engine = MemoryEngine(self.db)
            mem_results = await engine.recall(
                query=query, tenant_id=tenant_id, top_k=5
            )
            memory_results = [
                MemorySearchResult(
                    memory=MemoryResponse.model_validate(m), score=s
                )
                for m, s in mem_results
            ]

        graph_context = None
        if include_graph:
            try:
                from neuralake.core.knowledge_graph.engine import KnowledgeGraphEngine

                kg = KnowledgeGraphEngine(self.db)
                graph_context = await kg.get_context(query, tenant_id)
            except Exception:
                pass

        return SearchResult(
            chunks=chunks,
            memories=memory_results,
            graph_context=graph_context,
        )
