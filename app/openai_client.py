from __future__ import annotations

from typing import Dict, List

from openai import OpenAI

from app.config import settings

_client: OpenAI | None = None


def get_openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def call_gpt(messages: List[Dict[str, str]]) -> str:
    """
    Call OpenAI chat completion.

    Model is set via OPENAI_MODEL env var (default: gpt-4o-mini).
    User's intended model: gpt-5-nano — update OPENAI_MODEL in .env when available.
    """
    client = get_openai()
    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        max_completion_tokens=600,
    )
    return (resp.choices[0].message.content or "").strip()
