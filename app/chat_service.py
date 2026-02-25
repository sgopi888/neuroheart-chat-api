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
from app.token_budget import MAX_TOKENS, count_tokens, trim_text_to_tokens

logger = logging.getLogger(__name__)

# How many recent turns to keep verbatim
_RECENT_TURNS = 20
# Summarize when total messages exceed this threshold
_SUMMARIZE_THRESHOLD = 40
# Also trigger when older messages exceed this many tokens
_HISTORY_TOKEN_TRIGGER = 60_000
# Max tokens per RAG chunk
_RAG_CHUNK_TOKENS = 300
# Max RAG chunks
_RAG_MAX_CHUNKS = 20
# Max tokens for a summary produced by GPT
_SUMMARY_MAX_TOKENS = 800


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
    Trigger rolling summarization when:
      - total messages > _SUMMARIZE_THRESHOLD, OR
      - older messages (outside recent window) exceed _HISTORY_TOKEN_TRIGGER tokens.
    Returns the current rolling summary text.
    """
    total = count_messages(conversation_id)
    summary_row = get_or_create_summary(conversation_id, user_uid)
    current_summary = summary_row.get("summary") or ""
    last_summarized_id = summary_row.get("summarized_through_message_id")

    recent_ids = fetch_recent_message_ids(conversation_id, n=_RECENT_TURNS)
    if not recent_ids:
        return current_summary

    cutoff_id = recent_ids[0]  # oldest ID in the recent window

    if total <= _SUMMARIZE_THRESHOLD:
        # Check token-based trigger even when message count is low
        older = fetch_messages_for_summarization(
            conversation_id, after_id=last_summarized_id, before_id=cutoff_id
        )
        older_tokens = sum(count_tokens(m.get("content") or "") for m in older)
        if older_tokens < _HISTORY_TOKEN_TRIGGER:
            return current_summary
        to_summarize = older
    else:
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
        new_summary = trim_text_to_tokens(new_summary, _SUMMARY_MAX_TOKENS)
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
    user_message: str,
) -> tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Assemble prompt with strict tiktoken budget (MAX_TOKENS = 100k).

    Priority order:
      1. System prompt (always)
      2. Rolling summary
      3. HRV 14-day daily matrix
      4. HRV 90-day aggregates
      5. RAG snippets (≤20 chunks, ≤300 tokens each)
      6. Recent 20 messages
      7. User message (appended by caller)

    Returns (messages, token_breakdown).
    """
    messages: List[Dict[str, str]] = []
    breakdown: Dict[str, int] = {
        "tokens_system": 0,
        "tokens_summary": 0,
        "tokens_history": 0,
        "tokens_hrv": 0,
        "tokens_rag": 0,
        "tokens_total": 0,
    }

    # 1. System prompt — always
    sys_content = _system_prompt()
    messages.append({"role": "system", "content": sys_content})
    used = count_tokens(sys_content) + 4
    breakdown["tokens_system"] = used

    # 2. Rolling summary
    if summary:
        block = f"SESSION_SUMMARY:\n{summary}"
        tok = count_tokens(block) + 4
        messages.append({"role": "system", "content": block})
        used += tok
        breakdown["tokens_summary"] = tok

    # Pre-compute token costs for remaining blocks
    # HRV daily matrix
    hrv_daily_block = ""
    hrv_agg_block = ""
    if hrv_context:
        daily = hrv_context.get("daily_14d")
        if daily:
            hrv_daily_block = f"HRV_DAILY_14D:\n{json.dumps(daily, ensure_ascii=False)}"
        agg_keys = ("hrv_90d", "hr_90d", "sleep_90d", "steps_90d")
        agg = {k: hrv_context[k] for k in agg_keys if k in hrv_context}
        if agg:
            hrv_agg_block = f"HRV_AGGREGATES_90D:\n{json.dumps(agg, ensure_ascii=False)}"

    hrv_daily_tok = count_tokens(hrv_daily_block) + 4 if hrv_daily_block else 0
    hrv_agg_tok = count_tokens(hrv_agg_block) + 4 if hrv_agg_block else 0

    # RAG chunks (trim each to 300 tokens)
    trimmed_snips: List[str] = []
    for h in rag_hits[:_RAG_MAX_CHUNKS]:
        text = (h.get("text") or "").strip()
        if text:
            trimmed_snips.append(trim_text_to_tokens(text, _RAG_CHUNK_TOKENS))

    # History blocks
    history_blocks: List[Dict[str, str]] = [
        {"role": m["role"], "content": m["content"]}
        for m in history[-_RECENT_TURNS:]
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    # Reserve tokens for user message (appended outside this function)
    user_tok = count_tokens(user_message) + 4

    # --- Budget enforcement ---
    # Fixed overhead already used
    # Try to fit all; reduce RAG then history if over budget

    def _rag_tok(n: int) -> int:
        if n == 0:
            return 0
        return count_tokens("\n- ".join(trimmed_snips[:n])) + 4

    def _hist_tok(n: int) -> int:
        return sum(count_tokens(b["content"]) + 4 for b in history_blocks[-n:]) if n > 0 else 0

    hrv_tok = hrv_daily_tok + hrv_agg_tok
    rag_n = len(trimmed_snips)
    hist_n = len(history_blocks)

    # Reduce RAG first: 20→10→5→0
    for candidate_rag in (rag_n, 10, 5, 0):
        candidate_rag = min(candidate_rag, rag_n)
        for candidate_hist in (hist_n, 12, 8, 4, 0):
            candidate_hist = min(candidate_hist, hist_n)
            total = used + hrv_tok + _rag_tok(candidate_rag) + _hist_tok(candidate_hist) + user_tok
            if total <= MAX_TOKENS:
                rag_n = candidate_rag
                hist_n = candidate_hist
                break
        else:
            continue
        break

    # 3. HRV daily matrix
    hrv_included_tok = 0
    if hrv_daily_block:
        messages.append({"role": "system", "content": hrv_daily_block})
        used += hrv_daily_tok
        hrv_included_tok += hrv_daily_tok
    # 4. HRV aggregates
    if hrv_agg_block:
        messages.append({"role": "system", "content": hrv_agg_block})
        used += hrv_agg_tok
        hrv_included_tok += hrv_agg_tok
    breakdown["tokens_hrv"] = hrv_included_tok

    # 5. RAG snippets
    if rag_n > 0:
        rag_block = "RAG_CONTEXT:\n- " + "\n- ".join(trimmed_snips[:rag_n])
        rt = _rag_tok(rag_n)
        messages.append({"role": "system", "content": rag_block})
        used += rt
        breakdown["tokens_rag"] = rt

    # 6. History turns
    final_history = history_blocks[-hist_n:] if hist_n > 0 else []
    messages.extend(final_history)
    ht = _hist_tok(hist_n)
    used += ht
    breakdown["tokens_history"] = ht

    breakdown["tokens_total"] = used + user_tok
    return messages, breakdown


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

    # Build prompt with tiktoken budget enforcement
    prompt, breakdown = _build_prompt(summary, history, hrv_context, rag_hits, user_message)
    # Append user message as the final turn
    prompt.append({"role": "user", "content": user_message})

    # Call OpenAI
    reply = await asyncio.to_thread(call_gpt, prompt)

    # Persist assistant turn
    insert_message(
        user_uid,
        conversation_id,
        role="assistant",
        content=reply,
        model=None,
        metadata={"hrv_range": hrv_range, "rag_k": rag_k},
    )

    latency_ms = int((time.time() - t0) * 1000)
    used_context = bool(hrv_context) or bool(rag_hits)

    logger.info(
        "chat tokens — total=%d system=%d summary=%d history=%d hrv=%d rag=%d latency_ms=%d",
        breakdown["tokens_total"],
        breakdown["tokens_system"],
        breakdown["tokens_summary"],
        breakdown["tokens_history"],
        breakdown["tokens_hrv"],
        breakdown["tokens_rag"],
        latency_ms,
    )

    return {
        "reply": reply,
        "used_context": used_context,
        "hrv_range": hrv_range,
        "rag_k": len(rag_hits),
        "latency_ms": latency_ms,
    }
