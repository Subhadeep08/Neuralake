# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Neuralake is an AI knowledge base with a three-tier memory layer (episodic, semantic, procedural), combining vector search (pgvector), BM25 (tsvector), and a knowledge graph (recursive CTEs) via Reciprocal Rank Fusion. PostgreSQL-first: one database handles relational data, vectors, full-text search, and graph storage — no separate vector DB or graph DB required (Neo4j is optional/disabled by default). FastAPI backend, MCP server for agentic tool access, and a Python SDK (sync + async clients).

## Commands

```bash
make dev          # run API with reload (uvicorn neuralake.api.app:create_app --factory)
make up / down     # docker compose up/down (postgres+pgvector, redis, api)
make logs          # tail api container logs
make test          # pytest tests/ -v
make lint          # ruff check src/ tests/  &&  mypy src/neuralake/
make format        # ruff format src/ tests/
make migrate       # alembic upgrade head
make seed          # python scripts/seed_data.py
make mcp           # run MCP server (python -m neuralake.mcp.server)
```

Run a single test: `python -m pytest tests/unit/test_fusion.py -v` or `-k <name>`.

Local dev without Docker needs Postgres (pgvector extension) and Redis reachable per `.env` (copy from `.env.example`); `pip install -e ".[dev]"` first.

## Architecture

**Layout**: `src/neuralake/{api,core,storage,sdk,mcp,workers,config}`.

- `api/` — FastAPI app. `app.py` wires middleware and routers under `/api/v1`. `routers/` are thin — they call into `core/` engines. `middleware/` handles auth, tenant RLS, rate limiting, request logging.
- `core/` — business logic, organized by concern:
  - `query/engine.py` — top-level RAG orchestrator: classifies query intent, calls `retrieval/pipeline.py`, builds a prompt from retrieved chunks + memories, generates an answer via the LLM registry, returns citations.
  - `retrieval/pipeline.py` — hybrid search: runs vector search and sparse (tsvector) search over chunks in parallel-ish, fuses them with `retrieval/fusion.py` (RRF), then optionally layers in memory recall (`memory/engine.py`) and knowledge-graph context (`knowledge_graph/engine.py`).
  - `memory/` — `engine.py` stores/recalls memories (embeds content, ranks recall by `similarity * 0.7 + decay * 0.3` using `temporal.py`'s decay function); `extractor.py` pulls memories out of conversations via LLM; `consolidator.py` runs periodic promotion/archival (driven by `workers/scheduler.py` on a cron from `MemorySettings.consolidation_cron`).
  - `knowledge_graph/` — `extractor.py` does LLM-based entity/relationship extraction on ingest; `engine.py` queries stored entities and traverses neighbors for query-time context.
  - `llm/` and `embeddings/` — provider abstraction via `registry.py` (`get_llm()` / `get_embedder()`), each caching a module-level singleton keyed off `Settings`. Providers: Gemini/Anthropic/OpenAI/local(Ollama) for LLM; Gemini/OpenAI/Cohere/local for embeddings (Cohere and local fall back to the OpenAI-compatible embedder implementation).
  - `ingestion/` — `parsers/` (text, extend for PDF/docx per pyproject deps) → `chunking/recursive.py` → `pipeline.py` orchestrates parse → chunk → embed → persist.
- `storage/` — SQLAlchemy async models (`models/`) and repositories (`repositories/`). `database.py` owns the engine/session singletons and `set_tenant_context()`, which issues `SET LOCAL app.current_tenant_id` per-request for Postgres Row-Level Security — every request path must call this (see `api/dependencies.py::set_tenant_rls`) before touching tenant-scoped tables.
- `mcp/server.py` — exposes 8 tools (`neuralake_add_memory`, `_search_memories`, `_query`, `_search_documents`, `_ingest_document`, `_graph_explore`, `_extract_memories`, `_list_collections`) over stdio, each opening its own DB session and dispatching into the same `core/` engines used by the API. Note it currently hardcodes a single tenant UUID rather than deriving tenant from request context — don't assume it's multi-tenant aware yet.
- `sdk/` — `client.py` / `async_client.py`, thin HTTP wrappers over the REST API for external Python consumers.
- `config/settings.py` — Pydantic `BaseSettings`, one nested settings class per concern (`database`, `redis`, `embedding`, `llm`, `memory`, `search`, `auth`, `mcp`), all env-driven via `NEURALAKE_<SECTION>__<FIELD>` (see `.env.example`). `get_settings()` is `lru_cache`d — settings are read once per process.

**Auth**: dual-mode — JWT bearer (`create_jwt`/`_decode_jwt` in `middleware/auth.py`, user-scoped) or `X-API-Key: nl_sk_...` (tenant-scoped via `APIKey` table, hashed with SHA-256 at rest). `get_auth_context` resolves either into a common `AuthContext(tenant_id, user_id, role, scopes)`.

**Multi-tenancy**: enforced at the Postgres level via RLS, not just application filtering — `tenant_id` predicates alone are not sufficient/complete without the RLS session variable being set.

**Search fusion weights** (`SearchSettings`: vector/sparse/graph/memory) are currently defined but RRF fusion in `retrieval/pipeline.py` combines only vector+sparse; graph and memory results are appended as separate result sections in the response rather than fused into a single ranked list — check current code before assuming the configured weights are applied.

## Migrations

`migrations/` is an Alembic environment (`env.py`) but has no versioned migration scripts yet — `init_db()` in `storage/database.py` currently creates tables directly from `Base.metadata` for dev/test. When adding schema changes, generate a real Alembic revision rather than relying on `create_all`.
