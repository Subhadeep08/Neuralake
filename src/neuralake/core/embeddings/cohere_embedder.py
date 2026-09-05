import cohere

from neuralake.config.settings import get_settings
from neuralake.core.embeddings.base import BaseEmbedder


class CohereEmbedder(BaseEmbedder):
    def __init__(self, model: str = "embed-english-v3.0"):
        settings = get_settings()
        self.model = model
        self.client = cohere.AsyncClientV2(api_key=settings.embedding.cohere_api_key)
        self._dim = 1024

    async def embed_text(self, text: str) -> list[float]:
        response = await self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_document",
            embedding_types=["float"],
        )
        return response.embeddings.float_[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        batch_size = 96
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
                embedding_types=["float"],
            )
            all_embeddings.extend(response.embeddings.float_)
        return all_embeddings

    @property
    def dimensions(self) -> int:
        return self._dim
