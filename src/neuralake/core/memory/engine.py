import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.core.embeddings.registry import get_embedder
from neuralake.core.memory.temporal import compute_decay_score
from neuralake.storage.models.memory import Memory
from neuralake.storage.repositories.vector import VectorRepository


class MemoryEngine:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.vector_repo = VectorRepository(db)
        self.embedder = get_embedder()

    async def store(
        self,
        tenant_id: uuid.UUID,
        content: str,
        memory_type: str = "episodic",
        user_id: uuid.UUID | None = None,
        category: str | None = None,
        importance: float = 0.5,
        source: str = "api",
        metadata: dict | None = None,
        steps: list[dict] | None = None,
    ) -> Memory:
        embedding = await self.embedder.embed_text(content)

        memory = Memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            category=category,
            importance=importance,
            confidence=1.0,
            source=source,
            embedding=embedding,
            metadata=metadata or {},
            valid_from=datetime.now(timezone.utc),
            steps=steps,
        )
        self.db.add(memory)
        await self.db.flush()
        return memory

    async def recall(
        self,
        query: str,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        memory_types: list[str] | None = None,
        top_k: int = 10,
    ) -> list[tuple[Memory, float]]:
        embedding = await self.embedder.embed_text(query)

        results = await self.vector_repo.search_memories(
            embedding=embedding,
            tenant_id=tenant_id,
            user_id=user_id,
            memory_types=memory_types,
            top_k=top_k * 2,
        )

        scored = []
        for memory, similarity in results:
            decay = compute_decay_score(memory)
            combined = similarity * 0.7 + decay * 0.3
            scored.append((memory, combined))

            memory.access_count += 1
            memory.last_accessed = datetime.now(timezone.utc)

        scored.sort(key=lambda x: x[1], reverse=True)
        await self.db.flush()
        return scored[:top_k]
