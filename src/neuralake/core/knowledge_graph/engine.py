import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.storage.repositories.graph import GraphEntityRepository


class KnowledgeGraphEngine:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.entity_repo = GraphEntityRepository(db)

    async def find_entity(
        self, name: str, tenant_id: uuid.UUID, entity_type: str | None = None
    ):
        return await self.entity_repo.find_by_name(name, tenant_id, entity_type)

    async def get_context(
        self,
        query: str,
        tenant_id: uuid.UUID,
        max_depth: int = 2,
        limit: int = 20,
    ) -> list[dict]:
        entity = await self.entity_repo.find_by_name(query, tenant_id)
        if not entity:
            return []

        return await self.entity_repo.get_neighbors(
            entity_id=entity.id,
            tenant_id=tenant_id,
            max_depth=max_depth,
            limit=limit,
        )
