"""
Layer 2 — Long-term user memory extraction and retrieval.

After each chat exchange, a background task extracts key facts (entities,
preferences, health decisions, patterns) from the conversation and stores
them as vectors in Qdrant for semantic retrieval.

At query time, retrieves the most relevant memories for the current context.

Cost controls:
- Uses separate model (OPENAI_MODEL_MEM0) for background LLM calls
- Daily cost cap (MEM0_MAX_COST) — stops extraction when exceeded
- Throttle: skips trivial/short exchanges
- Toggle: MEMORY_EXTRACTION_ENABLED=false disables entirely
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
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

# Throttle: skip extraction if user message is shorter than this
_MIN_USER_MSG_LENGTH = 40

_qdrant: QdrantClient | None = None
_openai: OpenAI | None = None

# ---- Daily cost tracking (in-memory, resets on restart) ----
_daily_cost_cents: float = 0.0
_daily_cost_date: str = ""

# Approximate costs per 1k tokens (gpt-5-nano pricing estimate)
_COST_PER_1K_INPUT = 0.0001   # $0.0001 per 1k input tokens
_COST_PER_1K_OUTPUT = 0.0004  # $0.0004 per 1k output tokens
_COST_PER_1K_EMBED = 0.00002  # $0.00002 per 1k tokens (text-embedding-3-small)


def _track_cost(input_tokens: int, output_tokens: int, embed_tokens: int = 0) -> None:
    """Track daily cost in dollars. Resets each day."""
    global _daily_cost_cents, _daily_cost_date

    today = time.strftime("%Y-%m-%d")
    if _daily_cost_date != today:
        _daily_cost_cents = 0.0
        _daily_cost_date = today

    cost = (
        (input_tokens / 1000) * _COST_PER_1K_INPUT
        + (output_tokens / 1000) * _COST_PER_1K_OUTPUT
        + (embed_tokens / 1000) * _COST_PER_1K_EMBED
    )
    _daily_cost_cents += cost
    logger.debug("mem0 daily cost: $%.4f (added $%.6f)", _daily_cost_cents, cost)


def _is_over_daily_budget() -> bool:
    global _daily_cost_date, _daily_cost_cents
    today = time.strftime("%Y-%m-%d")
    if _daily_cost_date != today:
        _daily_cost_cents = 0.0
        _daily_cost_date = today
        return False
    return _daily_cost_cents >= settings.mem0_max_daily_cost


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
    _track_cost(0, 0, embed_tokens=resp.usage.total_tokens)
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
    """Use mem0 LLM model to extract key facts from a chat exchange."""
    prompt = _EXTRACT_PROMPT.format(user_msg=user_msg, assistant_msg=assistant_msg)
    resp = _get_openai().chat.completions.create(
        model=settings.openai_model_mem0,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=2048,
    )
    # Track cost
    if resp.usage:
        _track_cost(resp.usage.prompt_tokens, resp.usage.completion_tokens)

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return []

    try:
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
    """Retrieve relevant long-term memories for a user given current context."""
    try:
        client = _get_qdrant()
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
            if text and h.score >= 0.2:
                memories.append(text)
        return memories
    except Exception as exc:
        logger.warning("Memory retrieval failed: %s", exc)
        return []


def _should_extract(user_msg: str, assistant_msg: str) -> bool:
    """Decide whether this exchange is worth extracting memories from.
    Skips trivial messages like greetings, short questions, etc."""
    if not settings.memory_extraction_enabled:
        return False
    if _is_over_daily_budget():
        logger.info("mem0 daily budget ($%.2f) exceeded — skipping extraction", settings.mem0_max_daily_cost)
        return False
    if len(user_msg.strip()) < _MIN_USER_MSG_LENGTH:
        return False
    # Skip if assistant reply is very short (probably a greeting or clarification)
    if len(assistant_msg.strip()) < 50:
        return False
    return True


async def extract_and_store_memories(
    user_uid: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """
    Background task: extract facts from a chat exchange and store them.
    Runs async so it doesn't block the response.

    Guards:
    - MEMORY_EXTRACTION_ENABLED=false → skip
    - Daily cost cap (MEM0_MAX_COST) → skip when exceeded
    - Trivial message filter → skip short/greeting messages
    """
    if not _should_extract(user_msg, assistant_msg):
        return
    try:
        facts = await asyncio.to_thread(_extract_facts, user_msg, assistant_msg)
        if facts:
            await asyncio.to_thread(_store_facts, user_uid, facts)
    except Exception as exc:
        logger.warning("Memory extraction failed: %s", exc)


# ---- Cross-chat user profile (per-user memory, last 30 days) ----
# Each session adds ~10-20 words. Profile is a compact list of one-liners.
# Old entries (>30 days) are pruned. Strictly per-user — never cross-user.

_CROSS_CHAT_MAX_LINES = 50  # max lines in profile
_CROSS_CHAT_MAX_AGE_DAYS = 30

_CROSS_CHAT_PROMPT = """\
Add ONE short line (10-20 words max) summarizing what was discussed in this chat session.
This line will be appended to the user's memory profile.

Format: "YYYY-MM-DD: <one-liner about what was discussed>"

CHAT EXCHANGE:
User: {user_msg}
Assistant: {assistant_msg}

Output ONLY the single dated line, nothing else.
Example: "2026-03-14: Discussed 3am wake-ups linked to work stress, suggested body scan before bed"
"""


def update_cross_chat_profile(
    user_uid: str,
    user_messages: List[str],
    assistant_messages: List[str],
    existing_profile: str,
) -> str:
    """Append a 10-20 word session summary line to the user's cross-chat profile.
    Prunes entries older than 30 days. Uses mem0 model. Strictly per-user."""
    if not settings.cross_chat_memory_enabled:
        return existing_profile
    if _is_over_daily_budget():
        logger.info("mem0 daily budget exceeded — skipping cross-chat profile update")
        return existing_profile

    # Use last user/assistant message from this exchange
    user_text = user_messages[-1] if user_messages else ""
    asst_text = assistant_messages[-1] if assistant_messages else ""
    if not user_text or not asst_text:
        return existing_profile

    prompt = _CROSS_CHAT_PROMPT.format(user_msg=user_text, assistant_msg=asst_text)
    resp = _get_openai().chat.completions.create(
        model=settings.openai_model_mem0,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=256,
    )
    if resp.usage:
        _track_cost(resp.usage.prompt_tokens, resp.usage.completion_tokens)

    new_line = (resp.choices[0].message.content or "").strip().strip('"')
    if not new_line:
        return existing_profile

    # Append new line to existing profile
    lines = [l.strip() for l in (existing_profile or "").split("\n") if l.strip()]
    lines.append(new_line)

    # Prune entries older than 30 days
    import datetime
    cutoff = (datetime.datetime.now() - datetime.timedelta(days=_CROSS_CHAT_MAX_AGE_DAYS)).strftime("%Y-%m-%d")
    pruned = []
    for line in lines:
        # Lines start with "YYYY-MM-DD:" — keep if date >= cutoff
        date_part = line[:10]
        if len(date_part) == 10 and date_part >= cutoff:
            pruned.append(line)
        elif len(date_part) != 10:
            # Keep non-dated lines (legacy)
            pruned.append(line)

    # Cap at max lines (keep most recent)
    if len(pruned) > _CROSS_CHAT_MAX_LINES:
        pruned = pruned[-_CROSS_CHAT_MAX_LINES:]

    return "\n".join(pruned)
