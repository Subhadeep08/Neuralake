import json
import logging
import os

from typing import Any

from google import genai
from google.genai import types

from neuralake.config.settings import get_settings
from neuralake.core.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        settings = get_settings()
        self.model = model or settings.llm.model
        resolved_key = (
            api_key
            or settings.llm.gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
        )
        self.client = genai.Client(api_key=resolved_key)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def generate_structured(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type="application/json",
        )
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
        except Exception as e:
            logger.warning(
                "Gemini JSON mode generation error: %s, falling back to prompt instructions", e
            )
            full_prompt = prompt + "\n\nRespond with valid JSON only, no other text."
            text = await self.generate(
                full_prompt, system=system, max_tokens=max_tokens, temperature=temperature
            )
            text = text.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            return {"data": parsed}
        except json.JSONDecodeError:
            logger.warning("Failed to parse Gemini JSON response: %s", text[:200])
            return {}
