import pymupdf4llm


class PDFParser:
    def parse(self, content: bytes) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as f:
            f.write(content)
            f.flush()
            return pymupdf4llm.to_markdown(f.name)
