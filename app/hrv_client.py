from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_MAX_DAILY_ROWS = 14
_DAILY_FIELDS = {"date", "rmssd", "sdnn", "mean_hr", "lf_hf_ratio"}


def _shape_daily_matrix(time_series: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return last 14 rows keeping only the relevant daily fields."""
    rows = time_series[-_MAX_DAILY_ROWS:]
    out = []
    for row in rows:
        entry: Dict[str, Any] = {f: row[f] for f in _DAILY_FIELDS if f in row}
        out.append(entry)
    return out


def _shape_90d_aggregates(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract 90-day aggregate scalars from summary_metrics / patterns.
    Produces hrv_90d, hr_90d, sleep_90d, steps_90d — no large arrays.
    """
    sm = data.get("summary_metrics") or {}
    patterns = data.get("patterns") or {}

    hrv_90d: Dict[str, Any] = {}
    if "mean_rmssd" in sm:
        hrv_90d["mean_rmssd"] = sm["mean_rmssd"]
    trend = sm.get("trend") or patterns.get("hrv_trend")
    if trend:
        hrv_90d["trend"] = trend

    hr_90d: Dict[str, Any] = {}
    for key in ("mean_hr", "hr_mean", "mean_heart_rate"):
        if key in sm:
            hr_90d["mean"] = sm[key]
            break
    if "hr_p10" in sm:
        hr_90d["p10"] = sm["hr_p10"]
    if "hr_p90" in sm:
        hr_90d["p90"] = sm["hr_p90"]

    sleep_90d: Dict[str, Any] = {}
    for key in ("mean_sleep_hours", "avg_sleep_hours", "sleep_mean"):
        if key in sm:
            sleep_90d["mean_hours"] = sm[key]
            break
    if "sleep_trend" in patterns:
        sleep_90d["trend"] = patterns["sleep_trend"]

    steps_90d: Dict[str, Any] = {}
    for key in ("mean_steps", "avg_steps", "steps_mean"):
        if key in sm:
            steps_90d["mean"] = sm[key]
            break
    if "steps_trend" in patterns:
        steps_90d["trend"] = patterns["steps_trend"]

    result: Dict[str, Any] = {}
    if hrv_90d:
        result["hrv_90d"] = hrv_90d
    if hr_90d:
        result["hr_90d"] = hr_90d
    if sleep_90d:
        result["sleep_90d"] = sleep_90d
    if steps_90d:
        result["steps_90d"] = steps_90d
    return result


async def fetch_hrv_context(user_uid: str, hrv_range: str) -> Dict[str, Any]:
    """
    Fetch HRV analysis from the HRV API.

    user_uid is the apple_sub / external user id (passed as user_id param).
    Returns a compact dict:
      - daily_14d: list of up to 14 rows with {date, rmssd, sdnn, mean_hr, lf_hf_ratio}
      - hrv_90d / hr_90d / sleep_90d / steps_90d: aggregate scalars only
    Returns {} on error.
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

    time_series = data.get("time_series") or []
    daily_14d = _shape_daily_matrix(time_series)
    aggregates = _shape_90d_aggregates(data)

    result: Dict[str, Any] = {}
    if daily_14d:
        result["daily_14d"] = daily_14d
    result.update(aggregates)
    return result
