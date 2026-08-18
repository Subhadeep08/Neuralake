import uuid
from typing import Any, Generic, TypeVar

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.storage.models.base import TenantBase

T = TypeVar("T", bound=TenantBase)


class BaseRepository(Generic[T]):
    def __init__(self, model: type[T], session: AsyncSession):
        self.model = model
        self.session = session

    async def create(self, **kwargs: Any) -> T:
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def get(self, id: uuid.UUID) -> T | None:
        return await self.session.get(self.model, id)

    async def list(
        self,
        tenant_id: uuid.UUID,
        offset: int = 0,
        limit: int = 20,
        **filters: Any,
    ) -> tuple[list[T], int]:
        query = select(self.model).where(self.model.tenant_id == tenant_id)
        for key, value in filters.items():
            if value is not None and hasattr(self.model, key):
                query = query.where(getattr(self.model, key) == value)

        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.session.execute(count_query)).scalar() or 0

        query = query.offset(offset).limit(limit).order_by(self.model.created_at.desc())
        result = await self.session.execute(query)
        return list(result.scalars().all()), total

    async def update(self, id: uuid.UUID, **kwargs: Any) -> T | None:
        instance = await self.get(id)
        if instance is None:
            return None
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        await self.session.flush()
        return instance

    async def delete(self, id: uuid.UUID) -> bool:
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        return result.rowcount > 0
