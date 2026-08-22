import re

import numpy as np
import tiktoken

from neuralake.config.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from neuralake.core.embeddings.registry import get_embedder


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        breakpoint_percentile: float = 25.0,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.breakpoint_percentile = breakpoint_percentile
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _token_count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _split_sentences(self, text: str) -> list[str]:
        splits = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in splits if s.strip()]

    async def chunk(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text.strip()] if text.strip() else []

        embeddings = await self.embedder.embed_batch(sentences)
        emb_array = np.array(embeddings)

        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = emb_array / norms

        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

        threshold = np.percentile(similarities, self.breakpoint_percentile)

        groups: list[list[str]] = [[sentences[0]]]
        for i, sent in enumerate(sentences[1:]):
            if similarities[i] < threshold:
                groups.append([sent])
            else:
                groups[-1].append(sent)

        chunks = []
        for group in groups:
            merged = " ".join(group)
            if self._token_count(merged) <= self.chunk_size:
                chunks.append(merged)
            else:
                from neuralake.core.ingestion.chunking.recursive import RecursiveChunker

                fallback = RecursiveChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunks.extend(fallback.chunk(merged))

        return [c for c in chunks if c]
