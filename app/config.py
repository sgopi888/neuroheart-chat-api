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
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    # User intended gpt-5-nano; override via OPENAI_MODEL env var when available
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    app_token: str = os.getenv("APP_TOKEN", "")


settings = Settings()
