import json
import logging
import uuid

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

logger = logging.getLogger(__name__)

app = Server("neuralake")


def _tool_defs() -> list[Tool]:
    return [
        Tool(
            name="neuralake_add_memory",
            description="Store a memory (episodic, semantic, or procedural)",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Memory content"},
                    "memory_type": {"type": "string", "enum": ["episodic", "semantic", "procedural"], "default": "episodic"},
                    "category": {"type": "string"},
                    "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="neuralake_search_memories",
            description="Search over stored memories semantically",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "memory_types": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="neuralake_query",
            description="Full RAG query with hybrid retrieval and citations",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection_ids": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 10},
                    "include_memories": {"type": "boolean", "default": True},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="neuralake_search_documents",
            description="Hybrid search over documents without LLM generation",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "collection_ids": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="neuralake_ingest_document",
            description="Ingest text content into a collection",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "collection_id": {"type": "string"},
                },
                "required": ["title", "content", "collection_id"],
            },
        ),
        Tool(
            name="neuralake_graph_explore",
            description="Explore the knowledge graph from an entity",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "max_depth": {"type": "integer", "default": 2},
                    "limit": {"type": "integer", "default": 20},
                },
                "required": ["entity_name"],
            },
        ),
        Tool(
            name="neuralake_extract_memories",
            description="Extract memories from conversation messages",
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["messages"],
            },
        ),
        Tool(
            name="neuralake_list_collections",
            description="List available document collections",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return _tool_defs()


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    from neuralake.storage.database import get_session_factory

    factory = get_session_factory()
    async with factory() as db:
        try:
            result = await _dispatch(name, arguments, db)
            await db.commit()
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        except Exception as e:
            await db.rollback()
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _dispatch(name: str, args: dict, db) -> dict:
    from neuralake.config.settings import get_settings

    settings = get_settings()
    tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

    if name == "neuralake_add_memory":
        from neuralake.core.memory.engine import MemoryEngine

        engine = MemoryEngine(db)
        mem = await engine.store(
            tenant_id=tenant_id,
            content=args["content"],
            memory_type=args.get("memory_type", "episodic"),
            category=args.get("category"),
            importance=args.get("importance", 0.5),
            source="mcp",
        )
        return {"id": str(mem.id), "content": mem.content, "type": mem.memory_type}

    elif name == "neuralake_search_memories":
        from neuralake.core.memory.engine import MemoryEngine

        engine = MemoryEngine(db)
        results = await engine.recall(
            query=args["query"],
            tenant_id=tenant_id,
            memory_types=args.get("memory_types"),
            top_k=args.get("top_k", 10),
        )
        return {
            "memories": [
                {"content": m.content, "type": m.memory_type, "score": s}
                for m, s in results
            ]
        }

    elif name == "neuralake_query":
        from neuralake.core.query.engine import QueryEngine
        from neuralake.storage.repositories.vector import VectorRepository

        vector_repo = VectorRepository(db)
        engine = QueryEngine(vector_repo=vector_repo, db=db)
        collection_ids = [uuid.UUID(c) for c in args.get("collection_ids", [])] or None
        result = await engine.query(
            query=args["query"],
            tenant_id=tenant_id,
            collection_ids=collection_ids,
            top_k=args.get("top_k", 10),
            include_memories=args.get("include_memories", True),
        )
        return {"answer": result.answer, "intent": result.query_intent}

    elif name == "neuralake_search_documents":
        from neuralake.core.retrieval.pipeline import RetrievalPipeline
        from neuralake.storage.repositories.vector import VectorRepository

        vector_repo = VectorRepository(db)
        pipeline = RetrievalPipeline(vector_repo=vector_repo, db=db)
        collection_ids = [uuid.UUID(c) for c in args.get("collection_ids", [])] or None
        result = await pipeline.search(
            query=args["query"],
            tenant_id=tenant_id,
            collection_ids=collection_ids,
            top_k=args.get("top_k", 10),
        )
        return {
            "chunks": [
                {"content": c.content, "score": c.score} for c in result.chunks
            ]
        }

    elif name == "neuralake_ingest_document":
        from neuralake.core.ingestion.pipeline import IngestionPipeline
        from neuralake.storage.models.document import Document
        from sqlalchemy import select

        from neuralake.storage.models.collection import Collection

        coll = await db.get(Collection, uuid.UUID(args["collection_id"]))
        if not coll:
            return {"error": "Collection not found"}

        doc = Document(
            tenant_id=tenant_id,
            collection_id=coll.id,
            title=args["title"],
            source_type="text",
            raw_size_bytes=len(args["content"].encode()),
            status="pending",
        )
        db.add(doc)
        await db.flush()

        pipeline = IngestionPipeline()
        count = await pipeline.process_text(
            text=args["content"],
            document_id=doc.id,
            collection=coll,
            tenant_id=tenant_id,
            db=db,
        )
        doc.status = "completed"
        doc.chunk_count = count
        return {"document_id": str(doc.id), "chunks_created": count}

    elif name == "neuralake_graph_explore":
        from neuralake.core.knowledge_graph.engine import KnowledgeGraphEngine

        kg = KnowledgeGraphEngine(db)
        results = await kg.get_context(
            query=args["entity_name"],
            tenant_id=tenant_id,
            max_depth=args.get("max_depth", 2),
            limit=args.get("limit", 20),
        )
        return {"neighbors": results}

    elif name == "neuralake_extract_memories":
        from neuralake.core.memory.extractor import MemoryExtractor

        extractor = MemoryExtractor()
        memories = await extractor.extract(
            messages=args["messages"],
            tenant_id=tenant_id,
            user_id=None,
            db=db,
        )
        return {
            "extracted": [
                {"content": m.content, "type": m.memory_type} for m in memories
            ]
        }

    elif name == "neuralake_list_collections":
        from sqlalchemy import select

        from neuralake.storage.models.collection import Collection

        result = await db.execute(
            select(Collection).where(Collection.tenant_id == tenant_id).limit(50)
        )
        colls = result.scalars().all()
        return {
            "collections": [
                {"id": str(c.id), "name": c.name, "description": c.description}
                for c in colls
            ]
        }

    return {"error": f"Unknown tool: {name}"}


async def run_stdio():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())
