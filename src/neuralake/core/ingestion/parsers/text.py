from bs4 import BeautifulSoup, NavigableString, Tag


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


_HEADING_MAP = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


class HTMLParser:
    def parse(self, content: str | bytes) -> str:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")

        soup = BeautifulSoup(content, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        parts: list[str] = []
        self._walk(soup, parts)
        return "\n\n".join(p for p in parts if p.strip())

    def _walk(self, element: Tag, parts: list[str]) -> None:
        for child in element.children:
            if isinstance(child, NavigableString):
                text = child.strip()
                if text:
                    if parts:
                        parts[-1] = parts[-1] + " " + text if parts[-1] else text
                    else:
                        parts.append(text)
            elif isinstance(child, Tag):
                name = child.name
                if name in _HEADING_MAP:
                    parts.append(f"{_HEADING_MAP[name]} {child.get_text(strip=True)}")
                elif name == "p":
                    parts.append(child.get_text(strip=True))
                elif name in ("ul", "ol"):
                    for li in child.find_all("li", recursive=False):
                        parts.append(f"- {li.get_text(strip=True)}")
                elif name == "table":
                    self._parse_table(child, parts)
                elif name in ("pre", "code"):
                    parts.append(child.get_text())
                elif name in ("blockquote",):
                    parts.append(f"> {child.get_text(strip=True)}")
                else:
                    self._walk(child, parts)

    def _parse_table(self, table: Tag, parts: list[str]) -> None:
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            header_sep = "| " + " | ".join("---" for _ in rows[0].split("|")[1:-1]) + " |"
            rows.insert(1, header_sep)
            parts.append("\n".join(rows))
