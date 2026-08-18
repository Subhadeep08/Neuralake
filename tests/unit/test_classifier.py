import pytest

from neuralake.core.query.classifier import KEYWORD_PATTERNS


def test_keyword_patterns_exist():
    assert "lookup" in KEYWORD_PATTERNS
    assert "analytical" in KEYWORD_PATTERNS
    assert "conversational" in KEYWORD_PATTERNS
    assert "exploratory" in KEYWORD_PATTERNS


def test_lookup_keywords():
    patterns = KEYWORD_PATTERNS["lookup"]
    assert "what is" in patterns
    assert "who is" in patterns


def test_analytical_keywords():
    patterns = KEYWORD_PATTERNS["analytical"]
    assert "compare" in patterns
