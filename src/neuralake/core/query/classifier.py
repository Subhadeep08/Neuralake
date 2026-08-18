import logging

from neuralake.core.llm.prompts import (
    QUERY_CLASSIFICATION_PROMPT,
    QUERY_CLASSIFICATION_SYSTEM,
)
from neuralake.core.llm.registry import get_extraction_llm

logger = logging.getLogger(__name__)

KEYWORD_PATTERNS = {
    "lookup": ["what is", "who is", "define", "when was", "where is"],
    "analytical": ["compare", "difference between", "trend", "analysis", "why"],
    "conversational": ["tell me more", "thanks", "ok", "got it", "continue"],
    "exploratory": ["what can you", "tell me about", "explore", "overview"],
}


async def classify_query(query: str) -> tuple[str, float]:
    query_lower = query.lower().strip()

    for intent, patterns in KEYWORD_PATTERNS.items():
        for pattern in patterns:
            if query_lower.startswith(pattern):
                return intent, 0.8

    try:
        llm = get_extraction_llm()
        prompt = QUERY_CLASSIFICATION_PROMPT.format(query=query)
        result = await llm.generate_structured(prompt, system=QUERY_CLASSIFICATION_SYSTEM)
        return result.get("intent", "lookup"), result.get("confidence", 0.5)
    except Exception:
        return "lookup", 0.5
