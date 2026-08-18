MEMORY_EXTRACTION_SYSTEM = """You are a memory extraction system. Analyze conversations and extract structured memories.

For each memory, determine:
- type: "episodic" (events, interactions), "semantic" (facts, preferences, knowledge), or "procedural" (workflows, how-to steps)
- content: the memory content as a clear, standalone statement
- category: a short category label
- importance: 0.0-1.0 score based on how important/useful this memory is
- For procedural memories, include "steps" as a list of step objects

Return a JSON array of extracted memories."""

MEMORY_EXTRACTION_PROMPT = """Extract memories from this conversation:

{messages}

Return JSON array:
[{{"type": "episodic|semantic|procedural", "content": "...", "category": "...", "importance": 0.0-1.0, "steps": [...]}}]"""

ENTITY_EXTRACTION_SYSTEM = """You are a knowledge graph entity and relationship extractor. Given text, identify:
1. Named entities (people, organizations, products, technologies, concepts, locations, events)
2. Relationships between entities

Return structured JSON."""

ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from this text:

{text}

Return JSON:
{{
  "entities": [{{"name": "...", "type": "person|organization|product|technology|concept|location|event", "description": "..."}}],
  "relationships": [{{"source": "entity name", "target": "entity name", "relation": "relation_type", "weight": 0.0-1.0}}]
}}"""

QUERY_CLASSIFICATION_SYSTEM = """You are a query intent classifier. Classify queries into one of:
- lookup: Simple factual lookup ("What is X?", "Who is Y?")
- analytical: Requires analysis across multiple sources ("Compare X and Y", "What are the trends?")
- conversational: Casual or follow-up questions ("Tell me more", "Thanks")
- exploratory: Open-ended exploration ("What can you tell me about X?")"""

QUERY_CLASSIFICATION_PROMPT = """Classify this query intent:

Query: {query}

Return JSON: {{"intent": "lookup|analytical|conversational|exploratory", "confidence": 0.0-1.0}}"""

RAG_ANSWER_SYSTEM = """You are a knowledgeable AI assistant answering questions based on retrieved context.
Use the provided context to answer accurately. If the context doesn't contain enough information, say so.
Always cite your sources by referencing the chunk IDs provided."""

RAG_ANSWER_PROMPT = """Context:
{context}

{memory_context}

Question: {query}

Provide a comprehensive answer based on the context above. Reference specific sources using [chunk_id] notation."""

CONSOLIDATION_SYSTEM = """You are a memory consolidation system. Given a cluster of related episodic memories,
synthesize them into a single semantic memory that captures the essential pattern or fact."""

CONSOLIDATION_PROMPT = """Consolidate these related memories into a single semantic memory:

{memories}

Return JSON: {{"content": "consolidated fact/pattern", "category": "...", "importance": 0.0-1.0, "confidence": 0.0-1.0}}"""

CONTRADICTION_DETECTION_PROMPT = """Do these two memories contradict each other?

Memory 1: {memory1}
Memory 2: {memory2}

Return JSON: {{"contradicts": true/false, "explanation": "...", "keep": 1 or 2}}"""
