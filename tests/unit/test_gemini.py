from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import neuralake.core.embeddings.registry as embed_registry
import neuralake.core.llm.registry as llm_registry
from neuralake.config.settings import Settings
from neuralake.core.embeddings.gemini_embedder import GeminiEmbedder
from neuralake.core.embeddings.registry import get_embedder
from neuralake.core.llm.gemini import GeminiLLM
from neuralake.core.llm.registry import get_extraction_llm, get_llm


@pytest.mark.asyncio
async def test_gemini_llm_generate():
    with patch("neuralake.core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_generate_resp = MagicMock()
        mock_generate_resp.text = "Gemini generated response"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_generate_resp)

        llm = GeminiLLM(model="gemini-3.7-flash", api_key="test-api-key")
        result = await llm.generate(prompt="Hello", system="You are a helpful assistant")

        assert result == "Gemini generated response"
        mock_client.aio.models.generate_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-3.7-flash"
        assert call_kwargs["contents"] == "Hello"
        assert call_kwargs["config"].system_instruction == "You are a helpful assistant"


@pytest.mark.asyncio
async def test_gemini_llm_generate_structured():
    with patch("neuralake.core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_generate_resp = MagicMock()
        mock_generate_resp.text = '{"name": "Neuralake", "category": "database"}'
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_generate_resp)

        llm = GeminiLLM(model="gemini-3.5-flash-lite", api_key="test-api-key")
        result = await llm.generate_structured(prompt="Extract entity")

        assert result == {"name": "Neuralake", "category": "database"}
        call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["config"].response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_gemini_llm_generate_structured_markdown_fence():
    with patch("neuralake.core.llm.gemini.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_generate_resp = MagicMock()
        mock_generate_resp.text = '```json\n{"status": "ok"}\n```'
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_generate_resp)

        llm = GeminiLLM(model="gemini-3.7-flash", api_key="test-api-key")
        result = await llm.generate_structured(prompt="Status check")

        assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_gemini_embedder_embed_text():
    with patch("neuralake.core.embeddings.gemini_embedder.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        embedder = GeminiEmbedder(model="gemini-embedding-001", api_key="test-api-key")
        result = await embedder.embed_text("Sample query")

        assert len(result) == 1536
        assert embedder.dimensions == 1536
        mock_client.aio.models.embed_content.assert_awaited_once()
        call_kwargs = mock_client.aio.models.embed_content.call_args.kwargs
        assert call_kwargs["model"] == "gemini-embedding-001"
        assert call_kwargs["contents"] == "Sample query"
        assert call_kwargs["config"].output_dimensionality == 1536


@pytest.mark.asyncio
async def test_gemini_embedder_embed_batch():
    with patch("neuralake.core.embeddings.gemini_embedder.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_embedding1 = MagicMock()
        mock_embedding1.values = [0.1] * 1536
        mock_embedding2 = MagicMock()
        mock_embedding2.values = [0.2] * 1536
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding1, mock_embedding2]
        mock_client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        embedder = GeminiEmbedder(model="gemini-embedding-001", api_key="test-api-key")
        result = await embedder.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert len(result[0]) == 1536
        assert len(result[1]) == 1536


def test_gemini_registries():
    llm_registry._llm_instance = None
    llm_registry._extraction_llm = None
    embed_registry._embedder = None

    test_settings = Settings()
    test_settings.llm.provider = "gemini"
    test_settings.llm.model = "gemini-3.7-flash"
    test_settings.llm.extraction_model = "gemini-3.5-flash-lite"
    test_settings.embedding.provider = "gemini"
    test_settings.embedding.model = "gemini-embedding-001"

    with patch("neuralake.core.llm.registry.get_settings", return_value=test_settings), patch(
        "neuralake.core.embeddings.registry.get_settings", return_value=test_settings
    ), patch("neuralake.core.llm.gemini.genai.Client"), patch(
        "neuralake.core.embeddings.gemini_embedder.genai.Client"
    ):
        llm = get_llm()
        assert isinstance(llm, GeminiLLM)

        extraction_llm = get_extraction_llm()
        assert isinstance(extraction_llm, GeminiLLM)
        assert extraction_llm.model == "gemini-3.5-flash-lite"

        embedder = get_embedder()
        assert isinstance(embedder, GeminiEmbedder)
        assert embedder.dimensions == 1536
