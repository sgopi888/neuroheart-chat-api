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
    openai_embeddings_model: str = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    langsmith_tracing: bool = os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    lang_smith_key_legacy: str = os.getenv("LANG_SMITH_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "neuroheart-chat-api")
    langsmith_workspace_id: str = os.getenv("LANGSMITH_WORKSPACE_ID", "")
    app_token: str = os.getenv("APP_TOKEN", "")
    apple_bundle_id: str = os.getenv("APPLE_BUNDLE_ID", "com.neuroheart.app")
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
    # Audio / meditation settings
    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_space: str = os.getenv("HF_SPACE", "NeuroHeart2026/voice-agent")
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API", "")
    audio_storage_dir: str = os.getenv("AUDIO_STORAGE_DIR", "/opt/neuroheart/audio")
    audio_base_url: str = os.getenv("AUDIO_BASE_URL", "https://neuroheart.ai/audio")
    comfy_tts_url: str = os.getenv("COMFY_TTS_URL", "http://127.0.0.1:8844")
    comfy_tts_timeout: float = float(os.getenv("COMFY_TTS_TIMEOUT", "420"))

    # ── Hyperparameters (tunable, non-env) ──────────────────────────

    # Rate limiting
    rate_limit_capacity: float = 20.0
    rate_limit_refill_per_sec: float = 20.0 / 60.0

    # Token budget
    max_context_tokens: int = 100_000

    # Chat service
    chat_recent_turns: int = 15
    chat_summarize_threshold: int = 50
    chat_history_token_trigger: int = 50_000
    chat_rag_chunk_tokens: int = 300
    chat_rag_max_chunks: int = 20
    chat_summary_max_tokens: int = 800

    # HRV client
    hrv_max_daily_rows: int = 14
    hrv_client_timeout: float = 2.0

    # HRV Apple
    hrv_daily_window: int = 14
    hrv_agg_window: int = 90
    hrv_trend_threshold: float = 0.05

    # Ingest
    ingest_max_samples: int = 5000
    ingest_max_heartbeat_sessions: int = 5

    # Memory service
    memory_max_per_user: int = 200
    memory_retrieval_top_k: int = 5
    memory_min_msg_length: int = 40
    memory_duplicate_threshold: float = 0.92
    memory_cross_chat_max_lines: int = 50
    memory_cross_chat_max_age_days: int = 30

    # Meditation service
    meditation_music_length_ms: int = 60000
    meditation_max_stored: int = 25

    # RAG service
    rag_collection: str = "documents1"
    rag_max_passage_chars: int = 600

    # Auth
    auth_apple_keys_ttl: int = 86400

    # Mindfulness thresholds
    mindfulness_sdnn_threshold: int = 2
    mindfulness_rmssd_threshold: int = 3

    @property
    def database_url_psycopg(self) -> str:
        """Normalize SQLAlchemy-style URLs for direct psycopg connections."""
        return self.database_url.replace("+psycopg2", "")


settings = Settings()
