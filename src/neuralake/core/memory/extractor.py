import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.core.embeddings.registry import get_embedder
from neuralake.core.llm.prompts import MEMORY_EXTRACTION_PROMPT, MEMORY_EXTRACTION_SYSTEM
from neuralake.core.llm.registry import get_extraction_llm
from neuralake.storage.models.memory import Memory

logger = logging.getLogger(__name__)


class MemoryExtractor:
    async def extract(
        self,
        messages: list[dict],
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None,
        extract_types: list[str] | None = None,
        db: AsyncSession | None = None,
    ) -> list[Memory]:
        llm = get_extraction_llm()
        embedder = get_embedder()

        messages_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )
        prompt = MEMORY_EXTRACTION_PROMPT.format(messages=messages_text)
        result = await llm.generate_structured(prompt, system=MEMORY_EXTRACTION_SYSTEM)

        if not isinstance(result, list):
            result = result.get("memories", []) if isinstance(result, dict) else []

        memories = []
        for item in result:
            mem_type = item.get("type", "episodic")
            if extract_types and mem_type not in extract_types:
                continue

            content = item.get("content", "")
            if not content:
                continue

            embedding = await embedder.embed_text(content)

            memory = Memory(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_type=mem_type,
                content=content,
                category=item.get("category"),
                importance=item.get("importance", 0.5),
                confidence=1.0,
                source="extraction",
                embedding=embedding,
                valid_from=datetime.now(timezone.utc),
                steps=item.get("steps"),
            )

            if db:
                db.add(memory)
            memories.append(memory)

        if db:
            await db.flush()

        return memories
