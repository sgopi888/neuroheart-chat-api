from __future__ import annotations

import asyncio
from app.calendar_sync import format_calendar_context
import json
import logging
import time
from typing import Any, Dict, List

from app.history_repository import (
    count_messages,
    fetch_history,
    fetch_messages_for_summarization,
    fetch_recent_message_ids,
    get_cross_chat_profile,
    get_or_create_summary,
    insert_message,
    update_summary,
    upsert_cross_chat_profile,
)
from app.config import settings

if settings.hrv_local:
    from app.hrv_apple import fetch_hrv_context_apple as fetch_hrv_context
else:
    from app.hrv_client import fetch_hrv_context
from app.memory_service import extract_and_store_memories, retrieve_memories, update_cross_chat_profile
from app.openai_client import call_gpt, call_gpt_mem0
from app.rag_service import retrieve_rag
from app.token_budget import MAX_TOKENS, count_tokens, trim_text_to_tokens

logger = logging.getLogger(__name__)

# How many recent turns to keep verbatim (Layer 1 — short-term window)
_RECENT_TURNS = 10
# Summarize when total messages exceed this threshold (Layer 3 — rolling summary)
_SUMMARIZE_THRESHOLD = 50
# Trigger when older messages exceed this many tokens
_HISTORY_TOKEN_TRIGGER = 50_000
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
        "Use the context but be kind and like a counsellor and expert mindfulness expert. "
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
        new_summary = await asyncio.to_thread(call_gpt_mem0, prompt)
        new_summary = trim_text_to_tokens(new_summary, _SUMMARY_MAX_TOKENS)
        max_id = max(m["id"] for m in to_summarize)
        update_summary(conversation_id, max_id, new_summary)
        return new_summary
    except Exception as exc:
        logger.warning("Summarization failed: %s", exc)
        return current_summary


def _format_hrv_compact(hrv: Dict[str, Any]) -> str:
    """Convert HRV context dict to compact CSV-like string for minimal token usage."""
    parts: List[str] = []

    # Daily 14-day: date: (sdnn, hr)
    daily = hrv.get("daily_14d")
    if daily:
        rows = ", ".join(
            f"{d['date']}: ({d.get('sdnn', '-')}, {d.get('mean_hr', '-')})"
            for d in daily
        )
        parts.append(f"HRV_DAILY_14D (date: sdnn_ms, hr_bpm):\n{rows}")

    # Daily 90-day: date: (sdnn, hr)
    daily_90 = hrv.get("daily_90d")
    if daily_90:
        rows = ", ".join(
            f"{d['date']}: ({d.get('sdnn', '-')}, {d.get('mean_hr', '-')})"
            for d in daily_90
        )
        parts.append(f"HRV_DAILY_90D (date: sdnn_ms, hr_bpm):\n{rows}")

    # Hourly HRV 30-day: group by date, compact windows
    hrv_hourly = hrv.get("hrv_daily_hourly_30d")
    if hrv_hourly:
        by_date: Dict[str, List[str]] = {}
        for h in hrv_hourly:
            d = h["date"]
            # Extract start hour from window like "12:00-14:00"
            window = h.get("window", "")
            hour = window.split(":")[0] if window else "?"
            by_date.setdefault(d, []).append(f"{hour}h:{h['avg_value']}")
        rows = "\n".join(f"{d}: {', '.join(vals)}" for d, vals in by_date.items())
        parts.append(f"HRV_HOURLY_30D (date: hour:sdnn_ms):\n{rows}")

    # Hourly SDNN 30-day: same compact format
    hrv_sdnn_hourly = hrv.get("hrv_sdnn_daily_hourly_30d")
    if hrv_sdnn_hourly:
        by_date2: Dict[str, List[str]] = {}
        for h in hrv_sdnn_hourly:
            d = h["date"]
            window = h.get("window", "")
            hour = window.split(":")[0] if window else "?"
            by_date2.setdefault(d, []).append(f"{hour}h:{h['avg_value']}")
        rows = "\n".join(f"{d}: {', '.join(vals)}" for d, vals in by_date2.items())
        parts.append(f"HRV_SDNN_HOURLY_30D (date: hour:sdnn_ms):\n{rows}")

    # 90-day aggregates: compact key=value
    agg_parts: List[str] = []
    hrv_90d = hrv.get("hrv_90d")
    if hrv_90d:
        agg_parts.append(f"hrv: mean_sdnn={hrv_90d.get('mean_sdnn')}, trend={hrv_90d.get('trend')}")
    hr_90d = hrv.get("hr_90d")
    if hr_90d:
        agg_parts.append(
            f"hr: mean={hr_90d.get('mean')}, p10={hr_90d.get('p10')}, p90={hr_90d.get('p90')}"
        )
    sleep_90d = hrv.get("sleep_90d")
    if sleep_90d:
        agg_parts.append(
            f"sleep: mean_hours={sleep_90d.get('mean_hours')}, trend={sleep_90d.get('trend')}"
        )
    steps_90d = hrv.get("steps_90d")
    if steps_90d:
        agg_parts.append(f"steps: mean={steps_90d.get('mean')}, trend={steps_90d.get('trend')}")
    if agg_parts:
        parts.append("HEALTH_90D:\n" + "; ".join(agg_parts))

    return "\n\n".join(parts)


def _build_prompt(
    summary: str,
    memories: List[str],
    cross_chat_profile: str,
    history: List[Dict[str, Any]],
    hrv_context: Dict[str, Any],
    rag_hits: List[Dict[str, Any]],
    user_message: str,
    calendar_context: str = "",
) -> tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Assemble prompt with strict tiktoken budget (MAX_TOKENS = 100k).

    Priority order:
      1. System prompt (always)
      2. Rolling summary (Layer 3)
      3. Long-term memories (Layer 2)
      4. HRV context (compact)
      5. RAG snippets (≤20 chunks, ≤300 tokens each)
      6. Recent messages (Layer 1 — short-term window)
      7. User message (appended by caller)

    Returns (messages, token_breakdown).
    """
    messages: List[Dict[str, str]] = []
    breakdown: Dict[str, int] = {
        "tokens_system": 0,
        "tokens_summary": 0,
        "tokens_memory": 0,
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

    # 4b. Calendar context
    if calendar_context:
        cal_tok = count_tokens(calendar_context) + 4
        messages.append({"role": "system", "content": calendar_context})
        used += cal_tok

    # 2. Rolling summary (Layer 3)
    if summary:
        block = f"SESSION_SUMMARY:\n{summary}"
        tok = count_tokens(block) + 4
        messages.append({"role": "system", "content": block})
        used += tok
        breakdown["tokens_summary"] = tok

    # 3. Long-term memories (Layer 2)
    mem_block = ""
    if memories:
        mem_block = "USER_MEMORIES:\n- " + "\n- ".join(memories)
    mem_tok = count_tokens(mem_block) + 4 if mem_block else 0

    # 3b. Cross-chat user profile
    profile_block = ""
    if cross_chat_profile:
        profile_block = f"USER_PROFILE:\n{cross_chat_profile}"
    profile_tok = count_tokens(profile_block) + 4 if profile_block else 0

    # Pre-compute HRV context as compact CSV-like strings (no repeated keys)
    hrv_block = ""
    if hrv_context:
        hrv_block = _format_hrv_compact(hrv_context)
    hrv_tok = count_tokens(hrv_block) + 4 if hrv_block else 0

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

    rag_n = len(trimmed_snips)
    hist_n = len(history_blocks)

    # Reduce RAG first: 20→10→5→0
    for candidate_rag in (rag_n, 10, 5, 0):
        candidate_rag = min(candidate_rag, rag_n)
        for candidate_hist in (hist_n, 12, 8, 4, 0):
            candidate_hist = min(candidate_hist, hist_n)
            total = used + mem_tok + profile_tok + hrv_tok + _rag_tok(candidate_rag) + _hist_tok(candidate_hist) + user_tok
            if total <= MAX_TOKENS:
                rag_n = candidate_rag
                hist_n = candidate_hist
                break
        else:
            continue
        break

    # 3. Long-term memories (Layer 2)
    if mem_block:
        messages.append({"role": "system", "content": mem_block})
        used += mem_tok
    breakdown["tokens_memory"] = mem_tok

    # 3b. Cross-chat user profile
    if profile_block:
        messages.append({"role": "system", "content": profile_block})
        used += profile_tok

    # 4. HRV context (single compact block)
        messages.append({"role": "system", "content": hrv_block})
        used += hrv_tok
    breakdown["tokens_hrv"] = hrv_tok

    # 5. RAG snippets (knowledge base)
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


async def _update_cross_chat_profile_bg(
    user_uid: str, user_msg: str, assistant_msg: str, existing_profile: str
) -> None:
    """Background task to update the cross-chat user profile.
    Skips trivial messages (short greetings etc.)."""
    if len(user_msg.strip()) < 40 or len(assistant_msg.strip()) < 50:
        return
    try:
        new_profile = await asyncio.to_thread(
            update_cross_chat_profile, user_uid, [user_msg], [assistant_msg], existing_profile
        )
        if new_profile and new_profile != existing_profile:
            await asyncio.to_thread(upsert_cross_chat_profile, user_uid, new_profile)
            logger.info("Updated cross-chat profile for user %s", user_uid[:12])
    except Exception as exc:
        logger.warning("Cross-chat profile update failed: %s", exc)


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

    # Parallel: history (sync) + HRV (async) + RAG (thread) + memories (thread)
    history = fetch_history(user_uid, conversation_id, limit=_RECENT_TURNS + 2)
    hrv_task = asyncio.create_task(
        fetch_hrv_context(user_uid, hrv_range)
        if not settings.hrv_local
        else fetch_hrv_context(user_uid, hrv_range, mode=settings.hrv_mode)
    )
    rag_hits = await asyncio.to_thread(retrieve_rag, user_message, user_uid, "documents1")

    # Layer 2: retrieve long-term memories (semantic search on user facts)
    memories = await asyncio.to_thread(retrieve_memories, user_uid, user_message)

    # Cross-chat user profile
    cross_chat_profile = ""
    if settings.cross_chat_memory_enabled:
        cross_chat_profile = await asyncio.to_thread(get_cross_chat_profile, user_uid)

    hrv_context = await hrv_task
    # Calendar context (from synced events)
    calendar_block = await asyncio.to_thread(format_calendar_context, user_uid)
    # Calendar context (from synced events)

    # Layer 3: summarize old messages if needed (runs in background of this request)
    summary = await _maybe_summarize(conversation_id, user_uid)

    # Build prompt with 3-layer memory architecture
    prompt, breakdown = _build_prompt(summary, memories, cross_chat_profile, history, hrv_context, rag_hits, user_message,
        calendar_context=calendar_block)
    # Append user message as the final turn
    prompt.append({"role": "user", "content": user_message})

    if settings.debug_prompt_context:
        try:
            prompt_json = json.dumps(prompt, ensure_ascii=False, default=str)
            if len(prompt_json) > settings.debug_prompt_max_chars:
                prompt_json = prompt_json[: settings.debug_prompt_max_chars] + "\n...[truncated]"
            logger.info(
                "llm prompt debug — conversation_id=%s user_uid=%s hrv_range=%s prompt=%s",
                conversation_id,
                user_uid,
                hrv_range,
                prompt_json,
            )
        except Exception as exc:
            logger.warning("Prompt debug logging failed: %s", exc)

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

    # Layer 2: extract and store memories in background (no latency hit)
    asyncio.create_task(extract_and_store_memories(user_uid, user_message, reply))

    # Cross-chat profile update in background
    if settings.cross_chat_memory_enabled:
        asyncio.create_task(
            _update_cross_chat_profile_bg(user_uid, user_message, reply, cross_chat_profile)
        )

    latency_ms = int((time.time() - t0) * 1000)
    used_context = bool(hrv_context) or bool(rag_hits)

    logger.info(
        "chat tokens — total=%d system=%d summary=%d memory=%d history=%d hrv=%d rag=%d latency_ms=%d",
        breakdown["tokens_total"],
        breakdown["tokens_system"],
        breakdown["tokens_summary"],
        breakdown.get("tokens_memory", 0),
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
