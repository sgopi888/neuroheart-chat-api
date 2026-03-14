from __future__ import annotations

import logging
from typing import Dict, List

from openai import OpenAI

from app.config import settings
from app.llm_observability import traceable_call, wrap_openai_client

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_openai() -> OpenAI:
    global _client
    if _client is None:
        _client = wrap_openai_client(OpenAI(api_key=settings.openai_api_key))
    return _client


def call_gpt(messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI chat completion.

    Model is set via OPENAI_MODEL env var (default: gpt-4o-mini).
    User's intended model: gpt-5-nano — update OPENAI_MODEL in .env when available.
    """
    @traceable_call(run_name="chat_completion")
    def _call(messages: List[Dict[str, str]]) -> str:
        client = get_openai()
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            max_completion_tokens=settings.max_completion_tokens,
        )
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            logger.warning(
                "GPT returned empty — finish_reason=%s usage=%s",
                choice.finish_reason,
                resp.usage,
            )
        return content

    return _call(messages)


def call_gpt_mem0(messages: List[Dict[str, str]]) -> str:
    """Call OpenAI with the mem0 model (for background tasks like summarization)."""
    client = get_openai()
    resp = client.chat.completions.create(
        model=settings.openai_model_mem0,
        messages=messages,
        max_completion_tokens=settings.max_completion_tokens,
    )
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    if not content:
        logger.warning(
            "GPT mem0 returned empty — finish_reason=%s usage=%s",
            choice.finish_reason,
            resp.usage,
        )
    return content
