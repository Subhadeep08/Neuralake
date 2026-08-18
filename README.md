# Neuralake

**AI Knowledge Base with Memory Layer**

Production-grade knowledge base that goes beyond simple RAG. Combines vector search, BM25, knowledge graphs, and a three-tier memory system (episodic, semantic, procedural) to deliver contextually aware information retrieval.

## Features

- **Hybrid Retrieval** - Vector search (pgvector) + BM25 (tsvector) + Knowledge Graph traversal with Reciprocal Rank Fusion
- **Memory Layer** - Episodic, semantic, and procedural memories with temporal decay and automatic consolidation
- **Knowledge Graph** - LLM-powered entity/relationship extraction stored in PostgreSQL with recursive CTE traversal
- **Multi-tenant** - Row-Level Security, JWT + API key authentication
- **MCP Integration** - 8 MCP tools for seamless agentic AI integration
- **Python SDK** - Sync and async clients for programmatic access

## Quick Start

```bash
# Clone and set up
cp .env.example .env
# Edit .env with your API keys

# Start with Docker
docker compose up -d

# Or run directly
pip install -e ".[dev]"
make dev
```

## API

```
POST /api/v1/collections              # Manage document collections
POST /api/v1/collections/{id}/documents  # Ingest documents
POST /api/v1/memories                  # Store memories
POST /api/v1/memories/search           # Search memories
POST /api/v1/memories/extract          # Extract memories from conversations
POST /api/v1/search                    # Hybrid search
POST /api/v1/query                     # Full RAG query with citations
GET  /api/v1/graph/entities            # Explore knowledge graph
```

## Architecture

PostgreSQL-first design: single database for relational data, vector embeddings (pgvector), full-text search (tsvector), and graph storage (recursive CTEs). Claude (Anthropic) as primary LLM with OpenAI/Ollama alternatives.

## License

MIT
