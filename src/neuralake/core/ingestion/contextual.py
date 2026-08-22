import logging

from neuralake.core.llm.anthropic import AnthropicLLM

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """<document>
{document_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the context, nothing else."""


async def generate_chunk_context(
    document_text: str, chunk_text: str, llm: AnthropicLLM | None = None
) -> str:
    if llm is None:
        llm = AnthropicLLM()

    prompt = CONTEXT_PROMPT.format(document_text=document_text, chunk_text=chunk_text)

    try:
        response = await llm.generate(prompt)
        return response.strip()
    except Exception as e:
        logger.warning("Context generation failed for chunk: %s", e)
        return ""


async def enrich_chunks_with_context(
    document_text: str, chunks: list[str], llm: AnthropicLLM | None = None
) -> list[tuple[str, str]]:
    if llm is None:
        llm = AnthropicLLM()

    results = []
    for chunk in chunks:
        context = await generate_chunk_context(document_text, chunk, llm)
        enriched = f"{context}\n\n{chunk}" if context else chunk
        results.append((enriched, chunk))

    return results
