from neuralake.config.settings import get_settings
from neuralake.core.embeddings.base import BaseEmbedder

_embedder: BaseEmbedder | None = None


def get_embedder() -> BaseEmbedder:
    global _embedder
    if _embedder:
        return _embedder

    settings = get_settings()
    provider = settings.embedding.provider

    if provider == "openai":
        from neuralake.core.embeddings.openai_embedder import OpenAIEmbedder

        _embedder = OpenAIEmbedder()
    elif provider == "cohere":
        from neuralake.core.embeddings.cohere_embedder import CohereEmbedder

        _embedder = CohereEmbedder()
    elif provider == "local":
        from neuralake.core.embeddings.local_embedder import LocalEmbedder

        _embedder = LocalEmbedder()
    else:
        from neuralake.core.embeddings.openai_embedder import OpenAIEmbedder

        _embedder = OpenAIEmbedder()

    return _embedder
