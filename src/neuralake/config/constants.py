EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "BAAI/bge-m3": 1024,
    "all-MiniLM-L6-v2": 384,
}

SUPPORTED_DOCUMENT_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "text",
    "text/markdown": "markdown",
    "text/html": "html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

MAX_CHUNK_SIZE = 2048
MIN_CHUNK_SIZE = 64
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64

MEMORY_TYPES = ("episodic", "semantic", "procedural")

ENTITY_TYPES = (
    "person",
    "organization",
    "concept",
    "product",
    "location",
    "event",
    "technology",
    "custom",
)

QUERY_INTENTS = ("lookup", "analytical", "conversational", "exploratory")
