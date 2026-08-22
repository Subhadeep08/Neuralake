import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.config.settings import get_settings
from neuralake.core.embeddings.registry import get_embedder
from neuralake.core.memory.temporal import compute_decay_score, reinforce_memory
from neuralake.storage.models.memory import Memory
from neuralake.storage.repositories.vector import VectorRepository

logger = logging.getLogger(__name__)


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
        event_time: datetime | None = None,
    ) -> Memory:
        embedding = await self.embedder.embed_text(content)

        settings = get_settings()
        if settings.memory.consolidation_enabled:
            await self._check_contradictions(
                content, embedding, tenant_id, user_id
            )

        now = datetime.now(timezone.utc)
        weibull_scale = 168.0 + (importance * 336.0)

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
            event_time=event_time or now,
            recorded_at=now,
            valid_from=now,
            weibull_scale=weibull_scale,
            weibull_shape=0.7,
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

            reinforce_memory(memory)

        scored.sort(key=lambda x: x[1], reverse=True)
        await self.db.flush()
        return scored[:top_k]

    async def _check_contradictions(
        self,
        content: str,
        embedding: list[float],
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
    ) -> None:
        try:
            results = await self.vector_repo.search_memories(
                embedding=embedding,
                tenant_id=tenant_id,
                user_id=user_id,
                top_k=5,
            )

            for existing, similarity in results:
                if similarity < 0.8 or existing.valid_until is not None:
                    continue

                from neuralake.core.llm.prompts import CONTRADICTION_DETECTION_PROMPT
                from neuralake.core.llm.registry import get_extraction_llm

                llm = get_extraction_llm()
                prompt = CONTRADICTION_DETECTION_PROMPT.format(
                    memory1=existing.content, memory2=content
                )
                result = await llm.generate_structured(prompt)

                if result and result.get("contradicts"):
                    existing.valid_until = datetime.now(timezone.utc)
                    logger.info(
                        "Memory %s superseded by contradiction: %s",
                        existing.id,
                        result.get("explanation", ""),
                    )
        except Exception as e:
            logger.warning("Contradiction detection failed (non-fatal): %s", e)
