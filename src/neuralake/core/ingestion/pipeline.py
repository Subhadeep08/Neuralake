import hashlib
import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from neuralake.core.embeddings.registry import get_embedder
from neuralake.core.ingestion.chunking.recursive import RecursiveChunker
from neuralake.core.ingestion.parsers.text import HTMLParser, MarkdownParser, TextParser
from neuralake.storage.models.collection import Collection
from neuralake.storage.models.document import Chunk

logger = logging.getLogger(__name__)

PARSERS = {
    "text": TextParser,
    "markdown": MarkdownParser,
    "html": HTMLParser,
}


class IngestionPipeline:
    def __init__(self):
        self.embedder = get_embedder()

    async def process_text(
        self,
        text: str,
        document_id: uuid.UUID,
        collection: Collection,
        tenant_id: uuid.UUID,
        db: AsyncSession,
    ) -> int:
        parser_cls = PARSERS.get(collection.chunking_strategy, TextParser)
        if hasattr(parser_cls, "parse"):
            parsed = parser_cls().parse(text)
        else:
            parsed = text

        chunker = RecursiveChunker(
            chunk_size=collection.chunk_size,
            chunk_overlap=collection.chunk_overlap,
        )
        chunks = chunker.chunk(parsed)

        if not chunks:
            return 0

        embeddings = await self.embedder.embed_batch(chunks)

        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        for idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = Chunk(
                tenant_id=tenant_id,
                document_id=document_id,
                collection_id=collection.id,
                content=chunk_text,
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
