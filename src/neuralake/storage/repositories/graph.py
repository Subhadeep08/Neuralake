import uuid

from sqlalchemy import select, text, and_
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.storage.models.graph import GraphEntity, GraphEdge, EntityMention
from neuralake.storage.repositories.base import BaseRepository


class GraphEntityRepository(BaseRepository[GraphEntity]):
    def __init__(self, session: AsyncSession):
        super().__init__(GraphEntity, session)

    @staticmethod
    def _escape_like(value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    async def find_by_name(
        self, name: str, tenant_id: uuid.UUID, entity_type: str | None = None
    ) -> GraphEntity | None:
        escaped = self._escape_like(name)
        query = select(GraphEntity).where(
            and_(
                GraphEntity.tenant_id == tenant_id,
                GraphEntity.name.ilike(f"%{escaped}%"),
            )
        )
        if entity_type:
            query = query.where(GraphEntity.entity_type == entity_type)
        result = await self.session.execute(query.limit(1))
        return result.scalar_one_or_none()

    async def get_neighbors(
        self,
        entity_id: uuid.UUID,
        tenant_id: uuid.UUID,
        max_depth: int = 2,
        relation_types: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict]:
        relation_filter = ""
        params: dict = {
            "entity_id": str(entity_id),
            "tenant_id": str(tenant_id),
            "max_depth": max_depth,
            "limit": limit,
        }

        if relation_types:
            relation_filter = "AND e.relation_type = ANY(:relation_types)"
            params["relation_types"] = relation_types

        query = text(f"""
            WITH RECURSIVE graph_traverse AS (
                SELECT
                    e.source_id, e.target_id, e.relation_type, e.weight,
                    1 as depth,
                    ARRAY[e.source_id] as path
                FROM graph_edges e
                WHERE e.tenant_id = :tenant_id
                AND (e.source_id = :entity_id OR e.target_id = :entity_id)
                AND (e.valid_until IS NULL)
                {relation_filter}

                UNION ALL

                SELECT
                    e.source_id, e.target_id, e.relation_type, e.weight,
                    gt.depth + 1,
                    gt.path || e.source_id
                FROM graph_edges e
                JOIN graph_traverse gt ON (
                    e.source_id = gt.target_id OR e.target_id = gt.source_id
                )
                WHERE gt.depth < :max_depth
                AND e.tenant_id = :tenant_id
                AND NOT (e.source_id = ANY(gt.path))
                AND (e.valid_until IS NULL)
                {relation_filter}
            )
            SELECT DISTINCT
                ge.id, ge.name, ge.entity_type, ge.properties,
                gt.relation_type, gt.weight, gt.depth
            FROM graph_traverse gt
            JOIN graph_entities ge ON (
                ge.id = CASE
                    WHEN gt.source_id = :entity_id THEN gt.target_id
                    ELSE gt.source_id
                END
            )
            WHERE ge.tenant_id = :tenant_id
            ORDER BY gt.depth, gt.weight DESC
            LIMIT :limit
        """)

        result = await self.session.execute(query, params)

        return [
            {
                "id": str(row.id),
                "name": row.name,
                "entity_type": row.entity_type,
                "properties": row.properties,
                "relation_type": row.relation_type,
                "weight": row.weight,
                "depth": row.depth,
            }
            for row in result.all()
        ]


class GraphEdgeRepository(BaseRepository[GraphEdge]):
    def __init__(self, session: AsyncSession):
        super().__init__(GraphEdge, session)


class EntityMentionRepository(BaseRepository[EntityMention]):
    def __init__(self, session: AsyncSession):
        super().__init__(EntityMention, session)
