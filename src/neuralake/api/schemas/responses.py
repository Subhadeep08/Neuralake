import uuid
from datetime import datetime

from pydantic import BaseModel

from neuralake.api.schemas.common import PaginatedResponse


class CollectionResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    embedding_model: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    metadata: dict | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CollectionListResponse(PaginatedResponse):
    items: list[CollectionResponse]


class DocumentResponse(BaseModel):
    id: uuid.UUID
    collection_id: uuid.UUID
    title: str
    source_type: str
    source_uri: str | None
    content_hash: str | None
    raw_size_bytes: int | None
    chunk_count: int
    status: str
    error_message: str | None
    metadata: dict | None
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(PaginatedResponse):
    items: list[DocumentResponse]


class ChunkResponse(BaseModel):
    id: uuid.UUID
    content: str
    chunk_index: int
    token_count: int
    score: float | None = None
    metadata: dict | None

    class Config:
        from_attributes = True


class MemoryResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None
    memory_type: str
    content: str
    category: str | None
    importance: float
    confidence: float
    source: str | None
    valid_from: datetime | None
    valid_until: datetime | None
    access_count: int
    last_accessed: datetime | None
    consolidated: bool
    steps: list[dict] | None
    metadata: dict | None
    created_at: datetime

    class Config:
        from_attributes = True


class MemoryListResponse(PaginatedResponse):
    items: list[MemoryResponse]


class MemorySearchResult(BaseModel):
    memory: MemoryResponse
    score: float


class SearchResult(BaseModel):
    chunks: list[ChunkResponse]
    memories: list[MemorySearchResult] | None = None
    graph_context: list[dict] | None = None


class Citation(BaseModel):
    document_id: uuid.UUID | None = None
    document_title: str | None = None
    chunk_id: uuid.UUID
    content_snippet: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] | None = None
    memories_used: list[MemorySearchResult] | None = None
    graph_context: list[dict] | None = None
    query_intent: str | None = None


class GraphEntityResponse(BaseModel):
    id: uuid.UUID
    name: str
    entity_type: str
    properties: dict | None
    description: str | None
    mention_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class GraphNeighborResponse(BaseModel):
    id: str
    name: str
    entity_type: str
    properties: dict | None
    relation_type: str
    weight: float
    depth: int


class TenantResponse(BaseModel):
    id: uuid.UUID
    name: str
    slug: str
    plan: str
    settings: dict | None
    created_at: datetime

    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    id: uuid.UUID
    email: str
    name: str | None
    role: str
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyResponse(BaseModel):
    id: uuid.UUID
    key_prefix: str
    raw_key: str | None = None
    scopes: list[str]
    expires_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    details: dict | None = None


class UsageStatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_memories: int
    total_entities: int
    events: list[dict]
