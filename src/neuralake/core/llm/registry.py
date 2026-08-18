from neuralake.config.settings import get_settings
from neuralake.core.llm.base import BaseLLM

_llm_instance: BaseLLM | None = None
_extraction_llm: BaseLLM | None = None


def get_llm(model_override: str | None = None) -> BaseLLM:
    global _llm_instance
    if _llm_instance and not model_override:
        return _llm_instance

    settings = get_settings()
    provider = settings.llm.provider

    if provider == "anthropic":
        from neuralake.core.llm.anthropic import AnthropicLLM
        llm = AnthropicLLM(model=model_override)
    elif provider == "openai":
        from neuralake.core.llm.openai_llm import OpenAILLM
        llm = OpenAILLM(model=model_override)
    else:
        from neuralake.core.llm.openai_llm import OpenAILLM
        llm = OpenAILLM(
            model=model_override or settings.llm.model,
            api_key="ollama",
        )

    if not model_override:
        _llm_instance = llm
    return llm


def get_extraction_llm() -> BaseLLM:
    global _extraction_llm
    if _extraction_llm:
        return _extraction_llm

    settings = get_settings()
    _extraction_llm = get_llm(model_override=settings.llm.extraction_model)
    return _extraction_llm
