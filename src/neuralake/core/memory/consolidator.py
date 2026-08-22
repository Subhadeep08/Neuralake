import logging
import uuid
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import AgglomerativeClustering
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

        clusters = self._cluster_by_embedding(episodic_memories)

        promoted = 0
        archived = 0

        embedder = get_embedder()
        llm = get_extraction_llm()

        for cluster_id, memories in clusters.items():
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
                        category=consolidated.get("category"),
                        importance=consolidated.get("importance", 0.7),
                        confidence=consolidated.get("confidence", 0.8),
                        source="consolidation",
                        embedding=embedding,
                        recorded_at=datetime.now(timezone.utc),
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

    def _cluster_by_embedding(
        self,
        memories: list[Memory],
        distance_threshold: float = 0.3,
    ) -> dict[int, list[Memory]]:
        embeddings = []
        valid_memories = []
        for mem in memories:
            if mem.embedding is not None:
                embeddings.append(mem.embedding)
                valid_memories.append(mem)

        if len(valid_memories) < 3:
            return {0: valid_memories} if valid_memories else {}

        X = np.array(embeddings)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        X = X / norms

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(X)

        clusters: dict[int, list[Memory]] = {}
        for label, mem in zip(labels, valid_memories):
            clusters.setdefault(int(label), []).append(mem)

        return clusters
