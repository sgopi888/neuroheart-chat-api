"""
Layer 2 — Long-term user memory extraction and retrieval.

After each chat exchange, a background task extracts key facts (entities,
preferences, health decisions, patterns) from the conversation and stores
them as vectors in Qdrant for semantic retrieval.

At query time, retrieves the most relevant memories for the current context.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings

logger = logging.getLogger(__name__)

_MEMORY_COLLECTION = "user_memories"
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536
_MAX_MEMORIES_PER_USER = 200
_RETRIEVAL_TOP_K = 5

_qdrant: QdrantClient | None = None
_openai: OpenAI | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return _qdrant


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=settings.openai_api_key)
    return _openai


def _embed(text: str) -> List[float]:
    resp = _get_openai().embeddings.create(model=_EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def _ensure_collection() -> None:
    """Create the user_memories collection if it doesn't exist."""
    client = _get_qdrant()
    collections = [c.name for c in client.get_collections().collections]
    if _MEMORY_COLLECTION not in collections:
        client.create_collection(
            collection_name=_MEMORY_COLLECTION,
            vectors_config=qm.VectorParams(
                size=_EMBEDDING_DIM,
                distance=qm.Distance.COSINE,
            ),
        )
        # Create payload indexes for filtering
        client.create_payload_index(
            collection_name=_MEMORY_COLLECTION,
            field_name="user_uid",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection: %s", _MEMORY_COLLECTION)


_EXTRACT_PROMPT = """\
Extract key facts from this chat exchange that would be useful to remember \
for future conversations with this user. Focus on:
- Health conditions, symptoms, diagnoses mentioned
- User preferences (exercise habits, sleep schedule, diet, meditation practice)
- Personal context (job, stress triggers, goals, family situation)
- Decisions made or commitments (e.g. "will try breathing exercises before bed")
- Patterns noticed (e.g. "HRV drops on work-from-home days")

Output ONLY a JSON array of strings, each a concise standalone fact.
If nothing worth remembering, output [].
Example: ["User practices meditation 3x/week", "Has insomnia on Sunday nights"]

CHAT EXCHANGE:
User: {user_msg}
Assistant: {assistant_msg}
"""


def _extract_facts(user_msg: str, assistant_msg: str) -> List[str]:
    """Use LLM to extract key facts from a chat exchange."""
    prompt = _EXTRACT_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
    resp = _get_openai().chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2048,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return []

    import json
    try:
        # Handle markdown-wrapped JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        facts = json.loads(content)
        if isinstance(facts, list):
            return [str(f).strip() for f in facts if f and str(f).strip()]
    except (json.JSONDecodeError, IndexError):
        logger.warning("Failed to parse extracted facts: %s", content[:200])
    return []


def _find_duplicates(
    client: QdrantClient, user_uid: str, fact: str, threshold: float = 0.92
) -> bool:
    """Check if a very similar memory already exists."""
    vec = _embed(fact)
    response = client.query_points(
        collection_name=_MEMORY_COLLECTION,
        query=vec,
        query_filter=qm.Filter(
            must=[
                qm.FieldCondition(
                    key="user_uid",
                    match=qm.MatchValue(value=user_uid),
                ),
            ]
        ),
        limit=1,
        with_payload=True,
    )
    hits = response.points if hasattr(response, "points") else response
    if hits and hits[0].score >= threshold:
        return True
    return False


def _store_facts(user_uid: str, facts: List[str]) -> int:
    """Store new facts as vectors in Qdrant. Returns count stored."""
    if not facts:
        return 0

    client = _get_qdrant()
    _ensure_collection()

    stored = 0
    for fact in facts:
        if _find_duplicates(client, user_uid, fact):
            logger.debug("Skipping duplicate memory: %s", fact[:60])
            continue

        vec = _embed(fact)
        point_id = str(uuid.uuid4())
        client.upsert(
            collection_name=_MEMORY_COLLECTION,
            points=[
                qm.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "user_uid": user_uid,
                        "text": fact,
                        "type": "memory",
                    },
                )
            ],
        )
        stored += 1

    if stored:
        logger.info("Stored %d new memories for user %s", stored, user_uid[:12])
    return stored


def retrieve_memories(
    user_uid: str,
    query_context: str,
    top_k: int = _RETRIEVAL_TOP_K,
) -> List[str]:
    """
    Retrieve relevant long-term memories for a user given current context.

    Args:
        user_uid: The user's ID
        query_context: Text to search against (typically last few messages)
        top_k: Number of memories to retrieve

    Returns:
        List of memory strings, most relevant first.
    """
    try:
        client = _get_qdrant()
        # Check collection exists
        collections = [c.name for c in client.get_collections().collections]
        if _MEMORY_COLLECTION not in collections:
            return []

        vec = _embed(query_context)
        response = client.query_points(
            collection_name=_MEMORY_COLLECTION,
            query=vec,
            query_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="user_uid",
                        match=qm.MatchValue(value=user_uid),
                    ),
                ]
            ),
            limit=top_k,
            with_payload=True,
        )
        hits = response.points if hasattr(response, "points") else response

        memories = []
        for h in hits:
            text = (h.payload or {}).get("text", "").strip()
            if text and h.score >= 0.2:  # minimum relevance threshold
                memories.append(text)
        return memories
    except Exception as exc:
        logger.warning("Memory retrieval failed: %s", exc)
        return []


async def extract_and_store_memories(
    user_uid: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """
    Background task: extract facts from a chat exchange and store them.
    Runs async so it doesn't block the response.
    """
    try:
        facts = await asyncio.to_thread(_extract_facts, user_msg, assistant_msg)
        if facts:
            await asyncio.to_thread(_store_facts, user_uid, facts)
    except Exception as exc:
        logger.warning("Memory extraction failed: %s", exc)
