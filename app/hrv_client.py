from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Max daily rows to include in prompt (14 days)
_MAX_DAILY_ROWS = 14


async def fetch_hrv_context(user_uid: str, hrv_range: str) -> Dict[str, Any]:
    """
    Fetch HRV analysis from the HRV API.

    user_uid here is the apple_sub / external user id that the HRV API knows as user_id.
    Returns a compact dict suitable for prompt injection, or {} on error.
    """
    url = f"{settings.hrv_api_url}/v1/hrv/analysis"
    params = {"user_id": user_uid, "range": hrv_range}
    headers = {"x-api-key": settings.hrv_api_key}
    timeout = httpx.Timeout(2.0, connect=1.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params, headers=headers)

        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("HRV fetch failed: %s", exc)
        return {}

    # Keep only what we need — limit time_series to last 14 daily rows
    time_series = data.get("time_series") or []
    time_series = time_series[-_MAX_DAILY_ROWS:]

    return {
        "summary_metrics": data.get("summary_metrics"),
        "time_series": time_series,
        "patterns": data.get("patterns"),
    }
