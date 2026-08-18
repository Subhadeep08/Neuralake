import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.storage.models.document import Chunk
from neuralake.storage.models.memory import Memory


class VectorRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def search_chunks(
        self,
        embedding: list[float],
        tenant_id: uuid.UUID,
        collection_ids: list[uuid.UUID] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        query = (
            select(
                Chunk,
                (1 - Chunk.embedding.cosine_distance(text(f"'{embedding_str}'::vector"))).label(
                    "score"
                ),
            )
            .where(Chunk.tenant_id == tenant_id)
            .where(Chunk.embedding.is_not(None))
        )

        if collection_ids:
            query = query.where(Chunk.collection_id.in_(collection_ids))

        query = query.order_by(
            Chunk.embedding.cosine_distance(text(f"'{embedding_str}'::vector"))
        ).limit(top_k)

        result = await self.session.execute(query)
        rows = result.all()
        return [(row[0], row[1]) for row in rows if row[1] >= min_score]

    async def search_memories(
        self,
        embedding: list[float],
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        memory_types: list[str] | None = None,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[Memory, float]]:
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        query = (
            select(
                Memory,
                (1 - Memory.embedding.cosine_distance(text(f"'{embedding_str}'::vector"))).label(
                    "score"
                ),
            )
            .where(Memory.tenant_id == tenant_id)
            .where(Memory.embedding.is_not(None))
            .where(Memory.superseded_by.is_(None))
        )

        if user_id:
            query = query.where(Memory.user_id == user_id)
        if memory_types:
            query = query.where(Memory.memory_type.in_(memory_types))

        query = query.order_by(
            Memory.embedding.cosine_distance(text(f"'{embedding_str}'::vector"))
        ).limit(top_k)

        result = await self.session.execute(query)
        rows = result.all()
        return [(row[0], row[1]) for row in rows if row[1] >= min_score]

    async def sparse_search_chunks(
        self,
        query_text: str,
        tenant_id: uuid.UUID,
        collection_ids: list[uuid.UUID] | None = None,
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        ts_query = text(
            """
            SELECT c.*, ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', :query)) as score
            FROM chunks c
            WHERE c.tenant_id = :tenant_id
            AND ts_rank(to_tsvector('english', c.content), plainto_tsquery('english', :query)) > 0
            """
            + (" AND c.collection_id = ANY(:collection_ids)" if collection_ids else "")
            + """
            ORDER BY score DESC
            LIMIT :top_k
            """
        )

        params: dict = {"query": query_text, "tenant_id": str(tenant_id), "top_k": top_k}
        if collection_ids:
            params["collection_ids"] = [str(c) for c in collection_ids]

        result = await self.session.execute(ts_query, params)
        rows = result.all()

        chunks_with_scores = []
        for row in rows:
            chunk = await self.session.get(Chunk, row.id)
            if chunk:
                chunks_with_scores.append((chunk, row.score))
        return chunks_with_scores
