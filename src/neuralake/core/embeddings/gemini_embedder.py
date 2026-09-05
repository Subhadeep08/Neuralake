import os

from google import genai
from google.genai import types

from neuralake.config.constants import EMBEDDING_DIMENSIONS
from neuralake.config.settings import get_settings
from neuralake.core.embeddings.base import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.embedding.model
        self._dimensions = EMBEDDING_DIMENSIONS.get(self.model, settings.embedding.dimensions)
        resolved_key = (
            api_key
            or settings.embedding.gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
        )
        self.client = genai.Client(api_key=resolved_key)
        self.batch_size = settings.embedding.batch_size

    async def embed_text(self, text: str) -> list[float]:
        config = types.EmbedContentConfig(output_dimensionality=self._dimensions)
        response = await self.client.aio.models.embed_content(
            model=self.model,
            contents=text,
            config=config,
        )
        if response.embeddings:
            return response.embeddings[0].values or []
        return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        config = types.EmbedContentConfig(output_dimensionality=self._dimensions)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = await self.client.aio.models.embed_content(
                model=self.model,
                contents=batch,
                config=config,
            )
            if response.embeddings:
                all_embeddings.extend([e.values or [] for e in response.embeddings])
        return all_embeddings

    @property
    def dimensions(self) -> int:
        return self._dimensions

