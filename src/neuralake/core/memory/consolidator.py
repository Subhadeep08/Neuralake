import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.core.embeddings.registry import get_embedder
from neuralake.core.llm.prompts import CONSOLIDATION_PROMPT, CONSOLIDATION_SYSTEM
from neuralake.core.llm.registry import get_extraction_llm
from neuralake.core.memory.temporal import compute_decay_score, should_archive
from neuralake.storage.models.memory import Memory

logger = logging.getLogger(__name__)


class MemoryConsolidator:
    async def consolidate(
        self,
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> dict:
        result = await db.execute(
            select(Memory).where(
                Memory.tenant_id == tenant_id,
                Memory.memory_type == "episodic",
                Memory.consolidated == False,
                Memory.superseded_by.is_(None),
            )
        )
        episodic_memories = list(result.scalars().all())

        if len(episodic_memories) < 3:
            return {"promoted": 0, "archived": 0, "total_reviewed": len(episodic_memories)}

        clusters = self._cluster_by_category(episodic_memories)

        promoted = 0
        archived = 0

        embedder = get_embedder()
        llm = get_extraction_llm()

        for category, memories in clusters.items():
            if len(memories) >= 3:
                mem_texts = "\n".join(f"- {m.content}" for m in memories)
                prompt = CONSOLIDATION_PROMPT.format(memories=mem_texts)
                consolidated = await llm.generate_structured(
                    prompt, system=CONSOLIDATION_SYSTEM
                )

                if consolidated and consolidated.get("content"):
                    embedding = await embedder.embed_text(consolidated["content"])
                    new_memory = Memory(
                        tenant_id=tenant_id,
                        memory_type="semantic",
                        content=consolidated["content"],
                        category=consolidated.get("category", category),
                        importance=consolidated.get("importance", 0.7),
                        confidence=consolidated.get("confidence", 0.8),
                        source="consolidation",
                        embedding=embedding,
                        valid_from=datetime.now(timezone.utc),
                    )
                    db.add(new_memory)
                    await db.flush()

                    for mem in memories:
                        mem.consolidated = True
                        mem.superseded_by = new_memory.id

                    promoted += 1

        for mem in episodic_memories:
            if should_archive(mem):
                mem.valid_until = datetime.now(timezone.utc)
                archived += 1

        await db.flush()

        return {
            "promoted": promoted,
            "archived": archived,
            "total_reviewed": len(episodic_memories),
        }

    def _cluster_by_category(self, memories: list[Memory]) -> dict[str, list[Memory]]:
        clusters: dict[str, list[Memory]] = {}
        for mem in memories:
            key = mem.category or "uncategorized"
            clusters.setdefault(key, []).append(mem)
        return clusters
