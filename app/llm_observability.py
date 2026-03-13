from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def configure_langsmith_env() -> bool:
    """
    Normalize LangSmith env vars for the SDK.

    Supports the user's existing LANG_SMITH_KEY alias while preferring the
    official LANGSMITH_* variable names.
    """
    api_key = settings.langsmith_api_key or settings.lang_smith_key_legacy
    if not settings.langsmith_tracing or not api_key:
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key

    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id

    return True


def wrap_openai_client(client: T) -> T:
    """Wrap the OpenAI client for LangSmith tracing when enabled."""
    if not configure_langsmith_env():
        return client

    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception as exc:
        logger.warning("LangSmith OpenAI wrapping unavailable: %s", exc)
        return client


def traceable_call(fn: Callable[..., T], run_name: str) -> Callable[..., T]:
    """Wrap a function in a LangSmith trace root when enabled."""
    if not configure_langsmith_env():
        return fn

    try:
        from langsmith import traceable

        return traceable(name=run_name)(fn)
    except Exception as exc:
        logger.warning("LangSmith traceable wrapper unavailable: %s", exc)
        return fn
