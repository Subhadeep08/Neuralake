import logging
import uuid

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.core.embeddings.registry import get_embedder
from neuralake.storage.models.graph import GraphEntity
from neuralake.storage.models.memory import Memory

logger = logging.getLogger(__name__)


class SleepTimeCompute:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedder = get_embedder()

    async def compute_cross_references(self, tenant_id: uuid.UUID) -> dict:
        memories = (
            await self.db.execute(
                select(Memory).where(
                    Memory.tenant_id == tenant_id,
                    Memory.valid_until.is_(None),
                    Memory.embedding.isnot(None),
                ).limit(500)
            )
        ).scalars().all()

        entities = (
            await self.db.execute(
                select(GraphEntity).where(
                    GraphEntity.tenant_id == tenant_id,
                    GraphEntity.embedding.isnot(None),
                ).limit(500)
            )
        ).scalars().all()

        links_created = 0

        for memory in memories:
            if memory.embedding is None:
                continue
            for entity in entities:
                if entity.embedding is None:
                    continue
                similarity = self._cosine_similarity(memory.embedding, entity.embedding)
                if similarity > 0.7:
                    await self._upsert_cross_ref(
                        tenant_id=tenant_id,
                        source_type="memory",
                        source_id=memory.id,
                        target_type="entity",
                        target_id=entity.id,
                        relation="relates_to",
                        score=similarity,
                    )
                    links_created += 1

        for i, m1 in enumerate(memories):
            for m2 in memories[i + 1 :]:
                if m1.embedding is None or m2.embedding is None:
                    continue
                similarity = self._cosine_similarity(m1.embedding, m2.embedding)
                if similarity > 0.75:
                    await self._upsert_cross_ref(
                        tenant_id=tenant_id,
                        source_type="memory",
                        source_id=m1.id,
                        target_type="memory",
                        target_id=m2.id,
                        relation="similar_to",
                        score=similarity,
                    )
                    links_created += 1

        await self.db.flush()
        logger.info("Sleep compute: %d cross-references created for tenant %s", links_created, tenant_id)
        return {"links_created": links_created}

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        import numpy as np

        a_arr = np.array(a)
        b_arr = np.array(b)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    async def _upsert_cross_ref(
        self,
        tenant_id: uuid.UUID,
        source_type: str,
        source_id: uuid.UUID,
        target_type: str,
        target_id: uuid.UUID,
        relation: str,
        score: float,
    ) -> None:
        await self.db.execute(
            text("""
                INSERT INTO cross_references (tenant_id, source_type, source_id, target_type, target_id, relation, score)
                VALUES (:tenant_id, :source_type, :source_id, :target_type, :target_id, :relation, :score)
                ON CONFLICT (tenant_id, source_type, source_id, target_type, target_id, relation)
                DO UPDATE SET score = :score, updated_at = NOW()
            """),
            {
                "tenant_id": str(tenant_id),
                "source_type": source_type,
                "source_id": str(source_id),
                "target_type": target_type,
                "target_id": str(target_id),
                "relation": relation,
                "score": score,
            },
        )
