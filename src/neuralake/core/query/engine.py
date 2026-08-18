import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.api.schemas.responses import (
    Citation,
    MemoryResponse,
    MemorySearchResult,
    QueryResponse,
)
from neuralake.core.llm.prompts import RAG_ANSWER_PROMPT, RAG_ANSWER_SYSTEM
from neuralake.core.llm.registry import get_llm
from neuralake.core.query.classifier import classify_query
from neuralake.core.retrieval.pipeline import RetrievalPipeline
from neuralake.storage.repositories.vector import VectorRepository


class QueryEngine:
    def __init__(self, vector_repo: VectorRepository, db: AsyncSession):
        self.retrieval = RetrievalPipeline(vector_repo=vector_repo, db=db)
        self.db = db

    async def query(
        self,
        query: str,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        collection_ids: list[uuid.UUID] | None = None,
        top_k: int = 10,
        include_citations: bool = True,
        include_memories: bool = True,
        include_graph: bool = True,
        conversation_history: list[dict] | None = None,
    ) -> QueryResponse:
        intent, confidence = await classify_query(query)

        search_result = await self.retrieval.search(
            query=query,
            tenant_id=tenant_id,
            collection_ids=collection_ids,
            top_k=top_k,
            include_memories=include_memories,
            include_graph=include_graph,
        )

        context_parts = []
        for chunk in search_result.chunks:
            context_parts.append(f"[{chunk.id}] {chunk.content}")

        memory_context = ""
        if search_result.memories:
            mem_parts = [f"- {m.memory.content}" for m in search_result.memories]
            memory_context = "Relevant memories:\n" + "\n".join(mem_parts)

        context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant context found."

        prompt = RAG_ANSWER_PROMPT.format(
            context=context,
            memory_context=memory_context,
            query=query,
        )

        llm = get_llm()
        answer = await llm.generate(prompt, system=RAG_ANSWER_SYSTEM)

        citations = None
        if include_citations and search_result.chunks:
            citations = [
                Citation(
                    chunk_id=chunk.id,
                    content_snippet=chunk.content[:200],
                    score=chunk.score or 0.0,
                )
                for chunk in search_result.chunks[:5]
            ]

        return QueryResponse(
            answer=answer,
            citations=citations,
            memories_used=search_result.memories,
            graph_context=search_result.graph_context,
            query_intent=intent,
        )
