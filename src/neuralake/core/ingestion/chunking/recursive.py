import tiktoken

from neuralake.config.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE


class RecursiveChunker:
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        encoding_name: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.get_encoding(encoding_name)

    def _token_count(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk(self, text: str) -> list[str]:
        if self._token_count(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        return self._split(text, self.SEPARATORS)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text] if text.strip() else []

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            tokens = self.encoder.encode(text)
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_size, len(tokens))
                chunk_text = self.encoder.decode(tokens[start:end])
                if chunk_text.strip():
                    chunks.append(chunk_text.strip())
                start = end - self.chunk_overlap
                if start >= end:
                    break
            return chunks

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if self._token_count(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    if self._token_count(current) <= self.chunk_size:
                        chunks.append(current.strip())
                    else:
                        chunks.extend(self._split(current, remaining_seps))
                current = part

        if current:
            if self._token_count(current) <= self.chunk_size:
                chunks.append(current.strip())
            else:
                chunks.extend(self._split(current, remaining_seps))

        return [c for c in chunks if c]
