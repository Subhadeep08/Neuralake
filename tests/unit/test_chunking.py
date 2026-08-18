import pytest

try:
    from neuralake.core.ingestion.chunking.recursive import RecursiveChunker
    _TIKTOKEN_AVAILABLE = True
except Exception:
    _TIKTOKEN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TIKTOKEN_AVAILABLE,
    reason="tiktoken encoding unavailable (network restricted)",
)


@pytest.fixture
def chunker():
    try:
        return RecursiveChunker(chunk_size=100)
    except Exception:
        pytest.skip("tiktoken encoding unavailable")


def test_short_text_not_chunked(chunker):
    result = chunker.chunk("Short text.")
    assert len(result) == 1
    assert result[0] == "Short text."


def test_empty_text(chunker):
    result = chunker.chunk("")
    assert result == []


def test_long_text_chunked():
    try:
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
    except Exception:
        pytest.skip("tiktoken encoding unavailable")
    text = "Word " * 100
    result = chunker.chunk(text.strip())
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) > 0


def test_paragraph_splitting():
    try:
        chunker = RecursiveChunker(chunk_size=20)
    except Exception:
        pytest.skip("tiktoken encoding unavailable")
    text = "First paragraph content here.\n\nSecond paragraph content here."
    result = chunker.chunk(text)
    assert len(result) >= 1
