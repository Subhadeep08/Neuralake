import httpx


class NeuralakeClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        self._client = httpx.Client(base_url=self.base_url, headers=headers, timeout=60.0)

    def _url(self, path: str) -> str:
        return f"/api/v1{path}"

    def health(self) -> dict:
        return self._client.get(self._url("/health")).json()

    def create_collection(self, name: str, **kwargs) -> dict:
        return self._client.post(self._url("/collections"), json={"name": name, **kwargs}).json()

    def list_collections(self, offset: int = 0, limit: int = 20) -> dict:
        return self._client.get(self._url("/collections"), params={"offset": offset, "limit": limit}).json()

    def ingest_document(self, collection_id: str, title: str, content: str, **kwargs) -> dict:
        return self._client.post(
            self._url(f"/collections/{collection_id}/documents"),
            json={"title": title, "content": content, **kwargs},
        ).json()

    def search(self, query: str, **kwargs) -> dict:
        return self._client.post(self._url("/search"), json={"query": query, **kwargs}).json()

    def query(self, query: str, **kwargs) -> dict:
        return self._client.post(self._url("/query"), json={"query": query, **kwargs}).json()

    def add_memory(self, content: str, memory_type: str = "episodic", **kwargs) -> dict:
        return self._client.post(
            self._url("/memories"),
            json={"content": content, "memory_type": memory_type, **kwargs},
        ).json()

    def search_memories(self, query: str, **kwargs) -> dict:
        return self._client.post(self._url("/memories/search"), json={"query": query, **kwargs}).json()

    def extract_memories(self, messages: list[dict], **kwargs) -> dict:
        return self._client.post(self._url("/memories/extract"), json={"messages": messages, **kwargs}).json()

    def list_entities(self, search: str | None = None, **kwargs) -> list:
        params = kwargs
        if search:
            params["search"] = search
        return self._client.get(self._url("/graph/entities"), params=params).json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
