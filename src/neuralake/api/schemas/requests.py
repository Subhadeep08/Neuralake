import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    embedding_model: str = "text-embedding-3-small"
    chunking_strategy: str = "recursive"
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)
    contextual_retrieval: bool = False
    metadata: dict | None = None


class UpdateCollectionRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    metadata: dict | None = None


class IngestDocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    content: str
    source_type: str = "text"
    source_uri: str | None = None
    metadata: dict | None = None


class IngestURLRequest(BaseModel):
    url: str
    title: str | None = None
    collection_id: uuid.UUID | None = None
    metadata: dict | None = None


class CreateMemoryRequest(BaseModel):
    memory_type: Literal["episodic", "semantic", "procedural"] = "episodic"
    content: str = Field(min_length=1, max_length=10000)
    category: str | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str | None = None
    source_ref: str | None = None
    metadata: dict | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    steps: list[dict] | None = None
    trigger_conditions: list[str] | None = None


class UpdateMemoryRequest(BaseModel):
    content: str | None = None
    category: str | None = None
    importance: float | None = None
    valid_until: datetime | None = None
    metadata: dict | None = None


class SearchMemoriesRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    user_id: uuid.UUID | None = None
    memory_types: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractMemoriesRequest(BaseModel):
    messages: list[dict] = Field(min_length=1)
    user_id: uuid.UUID | None = None
    extract_types: list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    collection_ids: list[uuid.UUID] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    include_memories: bool = True
    include_graph: bool = True


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    collection_ids: list[uuid.UUID] | None = None
    user_id: uuid.UUID | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    include_citations: bool = True
    include_memories: bool = True
    include_graph: bool = True
    conversation_history: list[dict] | None = None


class GraphTraversalRequest(BaseModel):
    entity_id: uuid.UUID
    max_depth: int = Field(default=2, ge=1, le=5)
    relation_types: list[str] | None = None
    limit: int = Field(default=50, ge=1, le=200)


class CreateTenantRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    slug: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    plan: str = "free"
    settings: dict | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(default="default", max_length=255)
    scopes: list[str] = Field(default_factory=lambda: ["read", "write"])
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class CreateUserRequest(BaseModel):
    email: str
    name: str | None = None
    password: str = Field(min_length=8)
    role: str = "user"
