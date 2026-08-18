class TextParser:
    def parse(self, content: str | bytes) -> str:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return content.strip()


class MarkdownParser:
    def parse(self, content: str | bytes) -> str:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        return content.strip()


class HTMLParser:
    def parse(self, content: str | bytes) -> str:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        import re

        text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
