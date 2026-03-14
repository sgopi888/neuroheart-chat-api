from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "")
    hrv_api_url: str = os.getenv("HRV_API_URL", "http://127.0.0.1:8002")
    hrv_api_key: str = os.getenv("HRV_API_KEY", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_top_k: int = int(os.getenv("QDRANT_TOP_K", "10"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    lang_smith_key_legacy: str = os.getenv("LANG_SMITH_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "neuroheart-chat-api")
    langsmith_workspace_id: str = os.getenv("LANGSMITH_WORKSPACE_ID", "")
    app_token: str = os.getenv("APP_TOKEN", "")
    hrv_local: bool = os.getenv("HRV_LOCAL", "true").lower() in ("true", "1", "yes")
    hrv_mode: str = os.getenv("HRV_MODE", "compact")  # compact | meditation | full
    max_completion_tokens: int = int(os.getenv("MAX_COMPLETION_TOKENS", "16384"))
    # Background / mem0 settings
    openai_model_mem0: str = os.getenv("OPENAI_MODEL_MEM0", "gpt-5-nano")
    token_limit_mem0: int = int(os.getenv("TOKEN_LIMIT_MEM0", "50000"))
    mem0_max_daily_cost: float = float(os.getenv("MEM0_MAX_COST", "2.0"))
    memory_extraction_enabled: bool = os.getenv("MEMORY_EXTRACTION_ENABLED", "true").lower() in ("true", "1", "yes")
    cross_chat_memory_enabled: bool = os.getenv("CROSS_CHAT_MEMORY_ENABLED", "true").lower() in ("true", "1", "yes")
    debug_prompt_context: bool = os.getenv("DEBUG_PROMPT_CONTEXT", "false").lower() in ("true", "1", "yes")
    debug_prompt_max_chars: int = int(os.getenv("DEBUG_PROMPT_MAX_CHARS", "200000"))


settings = Settings()
