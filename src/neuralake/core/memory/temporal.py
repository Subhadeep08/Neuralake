import math
from datetime import datetime, timezone

from neuralake.storage.models.memory import Memory


def compute_decay_score(memory: Memory) -> float:
    if not memory.created_at:
        return memory.importance

    now = datetime.now(timezone.utc)
    created = memory.created_at.replace(tzinfo=timezone.utc) if memory.created_at.tzinfo is None else memory.created_at
    age_hours = (now - created).total_seconds() / 3600.0

    decay = math.exp(-memory.decay_rate * age_hours / 24.0)

    access_boost = min(memory.access_count * 0.02, 0.3)

    recency_boost = 0.0
    if memory.last_accessed:
        last = memory.last_accessed.replace(tzinfo=timezone.utc) if memory.last_accessed.tzinfo is None else memory.last_accessed
        since_access_hours = (now - last).total_seconds() / 3600.0
        recency_boost = 0.2 * math.exp(-0.05 * since_access_hours)

    score = memory.importance * decay + access_boost + recency_boost
    return min(max(score, 0.0), 1.0)


def should_archive(memory: Memory, threshold: float = 0.05) -> bool:
    return compute_decay_score(memory) < threshold
