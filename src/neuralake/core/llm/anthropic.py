import json
import logging

import anthropic

from neuralake.config.settings import get_settings
from neuralake.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.llm.model
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or settings.llm.anthropic_api_key
        )

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text

    async def generate_structured(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict:
        full_prompt = prompt + "\n\nRespond with valid JSON only, no other text."
        text = await self.generate(full_prompt, system=system, max_tokens=max_tokens, temperature=temperature)

        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response: %s", text[:200])
            return {}
