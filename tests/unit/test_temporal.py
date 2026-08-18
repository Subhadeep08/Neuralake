from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from neuralake.core.memory.temporal import compute_decay_score, should_archive


def _make_memory(
    importance=0.5,
    decay_rate=0.01,
    access_count=0,
    created_hours_ago=24,
    last_accessed=None,
):
    mem = MagicMock()
    mem.importance = importance
    mem.decay_rate = decay_rate
    mem.access_count = access_count
    mem.created_at = datetime.now(timezone.utc) - timedelta(hours=created_hours_ago)
    mem.last_accessed = last_accessed
    return mem


def test_recent_memory_high_score():
    mem = _make_memory(importance=0.8, created_hours_ago=1)
    score = compute_decay_score(mem)
    assert score > 0.5


def test_old_memory_low_score():
    mem = _make_memory(importance=0.3, created_hours_ago=720, decay_rate=0.05)
    score = compute_decay_score(mem)
    assert score < 0.5


def test_frequently_accessed_boost():
    mem1 = _make_memory(access_count=0)
    mem2 = _make_memory(access_count=10)
    assert compute_decay_score(mem2) > compute_decay_score(mem1)


def test_should_archive_old_low_importance():
    mem = _make_memory(importance=0.01, created_hours_ago=2000, decay_rate=0.1)
    assert should_archive(mem, threshold=0.05)


def test_should_not_archive_recent():
    mem = _make_memory(importance=0.8, created_hours_ago=1)
    assert not should_archive(mem)
