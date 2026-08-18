from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_DATABASE__")
    url: str = "postgresql+asyncpg://neuralake:neuralake@localhost:5432/neuralake"
    pool_size: int = 20
    max_overflow: int = 10
    echo: bool = False


class Neo4jSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_NEO4J__")
    enabled: bool = False
    url: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "neo4j123"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_REDIS__")
    url: str = "redis://localhost:6379/0"
    enabled: bool = True


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_EMBEDDING__")
    provider: Literal["openai", "cohere", "local"] = "openai"
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    openai_api_key: str | None = None
    cohere_api_key: str | None = None
    local_model_name: str = "BAAI/bge-m3"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_LLM__")
    provider: Literal["anthropic", "openai", "local"] = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    extraction_model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.1
    max_tokens: int = 4096
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    local_base_url: str = "http://localhost:11434"


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_MEMORY__")
    consolidation_enabled: bool = True
    consolidation_cron: str = "0 3 * * *"
    decay_rate_default: float = 0.01
    promotion_threshold_access_count: int = 5
    promotion_threshold_cluster_size: int = 3
    archive_threshold: float = 0.05
    max_memories_per_user: int = 10000


class SearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_SEARCH__")
    default_top_k: int = 10
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    vector_weight: float = 0.4
    sparse_weight: float = 0.2
    graph_weight: float = 0.2
    memory_weight: float = 0.2


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_AUTH__")
    jwt_secret: str = "CHANGE-ME-IN-PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    api_key_prefix: str = "nl_sk_"


class MCPSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEURALAKE_MCP__")
    enabled: bool = True
    transport: Literal["stdio", "sse", "streamable_http"] = "streamable_http"
    port: int = 3100


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEURALAKE_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = "Neuralake"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
