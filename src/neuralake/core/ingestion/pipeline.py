import hashlib
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.core.embeddings.registry import get_embedder
from neuralake.core.ingestion.chunking.recursive import RecursiveChunker
from neuralake.core.ingestion.parsers.docx import DOCXParser
from neuralake.core.ingestion.parsers.pdf import PDFParser
from neuralake.core.ingestion.parsers.text import HTMLParser, MarkdownParser, TextParser
from neuralake.storage.models.collection import Collection
from neuralake.storage.models.document import Chunk, Document

logger = logging.getLogger(__name__)

PARSERS = {
    "text": TextParser,
    "markdown": MarkdownParser,
    "html": HTMLParser,
    "pdf": PDFParser,
    "docx": DOCXParser,
}

CHUNKERS = {
    "recursive": RecursiveChunker,
}


def _get_chunker(strategy: str, chunk_size: int, chunk_overlap: int):
    chunker_cls = CHUNKERS.get(strategy, RecursiveChunker)
    return chunker_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class IngestionPipeline:
    def __init__(self):
        self.embedder = get_embedder()

    async def process(
        self,
        content: str | bytes,
        source_type: str,
        document_id: uuid.UUID,
        collection: Collection,
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> int:
        parser_cls = PARSERS.get(source_type, TextParser)
        parsed = parser_cls().parse(content)

        chunker = _get_chunker(
            collection.chunking_strategy,
            collection.chunk_size,
            collection.chunk_overlap,
        )

        if hasattr(chunker, "__await__") or hasattr(chunker.chunk, "__wrapped__"):
            chunks = await chunker.chunk(parsed)
        else:
            chunks = chunker.chunk(parsed)

        if not chunks:
            return 0

        enriched_chunks = chunks
        context_prefixes: list[str | None] = [None] * len(chunks)

        if collection.contextual_retrieval:
            try:
                from neuralake.core.ingestion.contextual import enrich_chunks_with_context

                pairs = await enrich_chunks_with_context(parsed, chunks)
                enriched_chunks = [p[0] for p in pairs]
                context_prefixes = [
                    p[0].replace(p[1], "").strip() or None for p in pairs
                ]
            except Exception as e:
                logger.warning("Contextual retrieval failed (using raw chunks): %s", e)

        embeddings = await self.embedder.embed_batch(enriched_chunks)

        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = Chunk(
                tenant_id=tenant_id,
                document_id=document_id,
                collection_id=collection.id,
                content=chunk_text,
                context_prefix=context_prefixes[idx],
                chunk_index=idx,
                token_count=len(enc.encode(chunk_text)),
                embedding=embedding,
                metadata={},
            )
            db.add(chunk)

        await db.flush()

        try:
            from neuralake.core.knowledge_graph.extractor import EntityExtractor

            extractor = EntityExtractor()
            await extractor.extract_from_chunks(chunks, tenant_id, db)
        except Exception as e:
            logger.warning("Entity extraction failed (non-fatal): %s", e)

        return len(chunks)

    async def process_text(
        self,
        text: str,
        document_id: uuid.UUID,
        collection: Collection,
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> int:
        return await self.process(
            content=text,
            source_type="text",
            document_id=document_id,
            collection=collection,
            tenant_id=tenant_id,
            db=db,
        )


def compute_content_hash(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


async def check_duplicate(
    content_hash: str,
    collection_id: uuid.UUID,
    tenant_id: uuid.UUID,
    db: AsyncSession,
) -> Document | None:
    result = await db.execute(
        select(Document).where(
            Document.tenant_id == tenant_id,
            Document.collection_id == collection_id,
            Document.content_hash == content_hash,
        ).limit(1)
    )
    return result.scalar_one_or_none()
