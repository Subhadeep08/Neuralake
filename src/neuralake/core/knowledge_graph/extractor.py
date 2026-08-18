import logging
import uuid

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.core.llm.prompts import ENTITY_EXTRACTION_PROMPT, ENTITY_EXTRACTION_SYSTEM
from neuralake.core.llm.registry import get_extraction_llm
from neuralake.core.embeddings.registry import get_embedder
from neuralake.storage.models.graph import GraphEdge, GraphEntity

logger = logging.getLogger(__name__)


class EntityExtractor:
    async def extract_from_chunks(
        self,
        chunks: list[str],
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> tuple[list[GraphEntity], list[GraphEdge]]:
        llm = get_extraction_llm()
        embedder = get_embedder()

        combined_text = "\n\n".join(chunks[:5])
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=combined_text)
        result = await llm.generate_structured(prompt, system=ENTITY_EXTRACTION_SYSTEM)

        entities_data = result.get("entities", [])
        relationships_data = result.get("relationships", [])

        entity_map: dict[str, GraphEntity] = {}
        created_entities: list[GraphEntity] = []

        for ent in entities_data:
            name = ent.get("name", "").strip()
            if not name:
                continue

            existing = await db.execute(
                select(GraphEntity).where(
                    and_(
                        GraphEntity.tenant_id == tenant_id,
                        GraphEntity.name == name,
                    )
                ).limit(1)
            )
            entity = existing.scalar_one_or_none()

            if entity:
                entity.mention_count += 1
                entity_map[name.lower()] = entity
            else:
                embedding = await embedder.embed_text(name)
                entity = GraphEntity(
                    tenant_id=tenant_id,
                    name=name,
                    entity_type=ent.get("type", "concept"),
                    description=ent.get("description"),
                    embedding=embedding,
                    mention_count=1,
                )
                db.add(entity)
                await db.flush()
                entity_map[name.lower()] = entity
                created_entities.append(entity)

        created_edges: list[GraphEdge] = []
        for rel in relationships_data:
            source_name = rel.get("source", "").strip().lower()
            target_name = rel.get("target", "").strip().lower()

            source = entity_map.get(source_name)
            target = entity_map.get(target_name)

            if source and target and source.id != target.id:
                edge = GraphEdge(
                    tenant_id=tenant_id,
                    source_id=source.id,
                    target_id=target.id,
                    relation_type=rel.get("relation", "related_to"),
                    weight=rel.get("weight", 0.5),
                )
                db.add(edge)
                created_edges.append(edge)

        await db.flush()
        return created_entities, created_edges
