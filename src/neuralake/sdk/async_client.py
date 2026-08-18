import httpx


class AsyncNeuralakeClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=60.0)

    def _url(self, path: str) -> str:
        return f"/api/v1{path}"

    async def health(self) -> dict:
        r = await self._client.get(self._url("/health"))
        return r.json()

    async def create_collection(self, name: str, **kwargs) -> dict:
        r = await self._client.post(self._url("/collections"), json={"name": name, **kwargs})
        return r.json()

    async def list_collections(self, offset: int = 0, limit: int = 20) -> dict:
        r = await self._client.get(self._url("/collections"), params={"offset": offset, "limit": limit})
        return r.json()

    async def ingest_document(self, collection_id: str, title: str, content: str, **kwargs) -> dict:
        r = await self._client.post(
            self._url(f"/collections/{collection_id}/documents"),
            json={"title": title, "content": content, **kwargs},
        )
        return r.json()

    async def search(self, query: str, **kwargs) -> dict:
        r = await self._client.post(self._url("/search"), json={"query": query, **kwargs})
        return r.json()

    async def query(self, query: str, **kwargs) -> dict:
        r = await self._client.post(self._url("/query"), json={"query": query, **kwargs})
        return r.json()

    async def add_memory(self, content: str, memory_type: str = "episodic", **kwargs) -> dict:
        r = await self._client.post(
            self._url("/memories"),
            json={"content": content, "memory_type": memory_type, **kwargs},
        )
        return r.json()

    async def search_memories(self, query: str, **kwargs) -> dict:
        r = await self._client.post(self._url("/memories/search"), json={"query": query, **kwargs})
        return r.json()

    async def extract_memories(self, messages: list[dict], **kwargs) -> dict:
        r = await self._client.post(self._url("/memories/extract"), json={"messages": messages, **kwargs})
        return r.json()

    async def list_entities(self, search: str | None = None, **kwargs) -> list:
        params = kwargs
        if search:
            params["search"] = search
        r = await self._client.get(self._url("/graph/entities"), params=params)
        return r.json()

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
