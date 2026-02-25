from __future__ import annotations

import logging
from typing import Any, Dict, List

import tiktoken

logger = logging.getLogger(__name__)

# cl100k_base covers gpt-4o, gpt-4o-mini, gpt-4, gpt-3.5-turbo
_ENC = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 100_000


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_ENC.encode(text))


def count_messages(messages: List[Dict[str, Any]]) -> int:
    """Count tokens across a list of chat messages (includes ~4 token role overhead per message)."""
    total = 0
    for m in messages:
        total += count_tokens(m.get("content") or "") + 4
    return total


def trim_text_to_tokens(text: str, max_tok: int) -> str:
    """Truncate text so it fits within max_tok tokens."""
    if not text:
        return text
    tokens = _ENC.encode(text)
    if len(tokens) <= max_tok:
        return text
    return _ENC.decode(tokens[:max_tok])
