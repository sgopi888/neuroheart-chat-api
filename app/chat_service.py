from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List

from app.history_repository import (
    count_messages,
    fetch_history,
    fetch_messages_for_summarization,
    fetch_recent_message_ids,
    get_or_create_summary,
    insert_message,
    update_summary,
)
from app.hrv_client import fetch_hrv_context
from app.openai_client import call_gpt
from app.rag_service import retrieve_rag

logger = logging.getLogger(__name__)

# Token budget: ~100k tokens × 4 chars/token
_CHAR_BUDGET = 400_000
# How many recent turns to keep verbatim
_RECENT_TURNS = 20
# Summarize when total messages exceed this threshold
_SUMMARIZE_THRESHOLD = 40


def _system_prompt() -> str:
    return (
        "You are NeuroHeart, a personal health insights assistant. "
        "Use the provided HRV and health context when relevant to give personalized advice. "
        "Never reference another user's data. "
        "Keep answers concise, practical, and supportive."
    )


def _summarization_prompt(existing_summary: str, messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant"):
            lines.append(f"{role.upper()}: {content}")
    history_text = "\n".join(lines)

    user_content = (
        "You are summarizing a health coaching chat session.\n\n"
    )
    if existing_summary:
        user_content += f"EXISTING SUMMARY:\n{existing_summary}\n\n"
    user_content += (
        f"NEW MESSAGES TO INCORPORATE:\n{history_text}\n\n"
        "Produce a concise updated summary in this structured format:\n"
        "- User profile / preferences learned:\n"
        "- Recurring symptoms / patterns mentioned:\n"
        "- Key events / dates:\n"
        "- Action items / commitments:\n"
        "- Open questions / unresolved threads:\n"
        "Keep each section to 1-3 bullet points maximum."
    )
    return [
        {"role": "system", "content": "You are a precise summarizer."},
        {"role": "user", "content": user_content},
    ]


async def _maybe_summarize(conversation_id: str, user_uid: str) -> str:
    """
    If message count exceeds threshold, summarize older messages.
    Returns the current rolling summary text.
    """
    total = count_messages(conversation_id)
    summary_row = get_or_create_summary(conversation_id, user_uid)
    current_summary = summary_row.get("summary") or ""
    last_summarized_id = summary_row.get("summarized_through_message_id")

    if total <= _SUMMARIZE_THRESHOLD:
        return current_summary

    # Find the oldest ID of the recent window we want to keep verbatim
    recent_ids = fetch_recent_message_ids(conversation_id, n=_RECENT_TURNS)
    if not recent_ids:
        return current_summary

    cutoff_id = recent_ids[0]  # oldest ID in the recent window

    # Fetch messages to summarize (older than the recent window)
    to_summarize = fetch_messages_for_summarization(
        conversation_id,
        after_id=last_summarized_id,
        before_id=cutoff_id,
    )

    if not to_summarize:
        return current_summary

    try:
        prompt = _summarization_prompt(current_summary, to_summarize)
        new_summary = await asyncio.to_thread(call_gpt, prompt)
        max_id = max(m["id"] for m in to_summarize)
        update_summary(conversation_id, max_id, new_summary)
        return new_summary
    except Exception as exc:
        logger.warning("Summarization failed: %s", exc)
        return current_summary


def _build_prompt(
    summary: str,
    history: List[Dict[str, Any]],
    hrv_context: Dict[str, Any],
    rag_hits: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Assemble prompt respecting character budget."""
    messages: List[Dict[str, str]] = []
    used_chars = 0
    budget = _CHAR_BUDGET

    # 1. System prompt (always included)
    sys_content = _system_prompt()
    messages.append({"role": "system", "content": sys_content})
    used_chars += len(sys_content)

    # 2. Rolling summary
    if summary:
        block = f"SESSION_SUMMARY:\n{summary}"
        messages.append({"role": "system", "content": block})
        used_chars += len(block)

    # 3. Recent messages (last _RECENT_TURNS, but budget-aware)
    history_msgs = history[-_RECENT_TURNS:]
    history_blocks: List[Dict[str, str]] = []
    for m in history_msgs:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            history_blocks.append({"role": role, "content": content})
    # Reserve space for history last
    history_char_est = sum(len(b["content"]) for b in history_blocks)

    # 4. HRV context
    if hrv_context and used_chars + 3000 < budget:
        hrv_json = json.dumps(hrv_context, ensure_ascii=False)[:3000]
        block = f"HRV_CONTEXT (JSON):\n{hrv_json}"
        messages.append({"role": "system", "content": block})
        used_chars += len(block)

    # 5. RAG snippets — reduce if over budget
    rag_chunks = rag_hits[:20]
    snips = [(h.get("text") or "").strip()[:400] for h in rag_chunks if (h.get("text") or "").strip()]
    rag_allowed = min(len(snips), 20)
    while rag_allowed > 0 and used_chars + sum(len(s) for s in snips[:rag_allowed]) + history_char_est > budget:
        rag_allowed -= 5
    if rag_allowed < 0:
        rag_allowed = 0
    if snips[:rag_allowed]:
        block = "RAG_CONTEXT (snippets):\n- " + "\n- ".join(snips[:rag_allowed])
        messages.append({"role": "system", "content": block})
        used_chars += len(block)

    # 6. Chat history turns
    allowed_turns = len(history_blocks)
    while allowed_turns > 0 and used_chars + sum(len(b["content"]) for b in history_blocks[:allowed_turns]) > budget:
        allowed_turns -= 4
    if allowed_turns < 0:
        allowed_turns = 0
    messages.extend(history_blocks[:allowed_turns])

    return messages


async def chat_once(
    user_uid: str,
    conversation_id: str,
    user_message: str,
    hrv_range: str,
    rag_k: int = 3,
) -> Dict[str, Any]:
    t0 = time.time()

    # Persist user turn
    insert_message(user_uid, conversation_id, role="user", content=user_message)

    # Parallel: history (sync) + HRV (async) + RAG (thread)
    history = fetch_history(user_uid, conversation_id, limit=_RECENT_TURNS + 2)
    hrv_task = asyncio.create_task(fetch_hrv_context(user_uid, hrv_range))
    rag_hits = await asyncio.to_thread(retrieve_rag, user_message, user_uid, "documents1", rag_k)
    hrv_context = await hrv_task

    # Summarize old messages if needed (runs in background of this request)
    summary = await _maybe_summarize(conversation_id, user_uid)

    # Build prompt
    prompt = _build_prompt(summary, history, hrv_context, rag_hits)

    # Call OpenAI
    reply = await asyncio.to_thread(call_gpt, prompt)

    # Persist assistant turn
    insert_message(
        user_uid,
        conversation_id,
        role="assistant",
        content=reply,
        model=None,  # model name is in settings
        metadata={"hrv_range": hrv_range, "rag_k": rag_k},
    )

    latency_ms = int((time.time() - t0) * 1000)
    used_context = bool(hrv_context) or bool(rag_hits)

    return {
        "reply": reply,
        "used_context": used_context,
        "hrv_range": hrv_range,
        "rag_k": len(rag_hits),
        "latency_ms": latency_ms,
    }
