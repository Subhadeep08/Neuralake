from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.api.middleware.auth import AuthContext, get_auth_context
from neuralake.storage.database import get_db
from neuralake.storage.repositories.base import BaseRepository
from neuralake.storage.repositories.graph import (
    EntityMentionRepository,
    GraphEdgeRepository,
    GraphEntityRepository,
)
from neuralake.storage.repositories.vector import VectorRepository
from neuralake.storage.models.collection import Collection
from neuralake.storage.models.document import Document
from neuralake.storage.models.memory import Memory


def get_collection_repo(db: AsyncSession = Depends(get_db)):
    return BaseRepository(Collection, db)


def get_document_repo(db: AsyncSession = Depends(get_db)):
    return BaseRepository(Document, db)


def get_memory_repo(db: AsyncSession = Depends(get_db)):
    return BaseRepository(Memory, db)


def get_vector_repo(db: AsyncSession = Depends(get_db)):
    return VectorRepository(db)


def get_graph_entity_repo(db: AsyncSession = Depends(get_db)):
    return GraphEntityRepository(db)


def get_graph_edge_repo(db: AsyncSession = Depends(get_db)):
    return GraphEdgeRepository(db)


def get_entity_mention_repo(db: AsyncSession = Depends(get_db)):
    return EntityMentionRepository(db)
