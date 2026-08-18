from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class EpisodicMemory:
    content: str
    category: str | None = None
    importance: float = 0.5
    source: str = "conversation"
    timestamp: datetime | None = None


@dataclass
class SemanticMemory:
    content: str
    category: str | None = None
    importance: float = 0.7
    confidence: float = 1.0
    source: str = "extraction"


@dataclass
class ProceduralMemory:
    content: str
    category: str | None = None
    importance: float = 0.6
    steps: list[dict] = field(default_factory=list)
    trigger_conditions: list[str] = field(default_factory=list)
