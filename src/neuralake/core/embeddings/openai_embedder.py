from openai import AsyncOpenAI

from neuralake.config.constants import EMBEDDING_DIMENSIONS
from neuralake.config.settings import get_settings
from neuralake.core.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.embedding.model
        self._dimensions = EMBEDDING_DIMENSIONS.get(self.model, 1536)
        self.client = AsyncOpenAI(api_key=api_key or settings.embedding.openai_api_key)
        self.batch_size = settings.embedding.batch_size

    async def embed_text(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            input=text, model=self.model
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = await self.client.embeddings.create(
                input=batch, model=self.model
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    @property
    def dimensions(self) -> int:
        return self._dimensions
