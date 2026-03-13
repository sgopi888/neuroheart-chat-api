"""
Server-side HRV analytics engine using NeuroKit2.

Replaces the external HRV API (hrv_client.py) by querying health_samples
directly from PostgreSQL and computing HRV metrics locally.

Data source tiers (highest priority first):
  Tier 1: heartbeat_series — RR intervals from payload JSONB
  Tier 2: hrv_sdnn — beat-to-beat BPM from payload JSONB → RR conversion
  Tier 3: hrv — Apple SDNN value only (current data)

Payload modes control how many metrics are included in LLM context:
  compact    — SDNN, RMSSD, LF/HF, mean_hr (default)
  meditation — top 10 meditation-research metrics
  full       — all computed metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text

from app.db import get_engine

logger = logging.getLogger(__name__)

# Temporary production mode: pass through Apple Health HRV only.
# Keep the tiered NeuroKit code in place for re-enabling later.
_APPLE_HEALTH_ONLY = True

# ---------------------------------------------------------------------------
# Optional NeuroKit2 import (only needed for Tier 1/2 with RR intervals)
# ---------------------------------------------------------------------------
try:
    import neurokit2 as nk
    import numpy as np

    _HAS_NEUROKIT = True
except ImportError:
    _HAS_NEUROKIT = False
    try:
        import numpy as np  # noqa: F811
    except ImportError:
        np = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RANGE_DAYS = {"1d": 1, "7d": 7, "30d": 30, "6m": 180}
_DAILY_WINDOW = 14
_AGG_WINDOW = 90
_TREND_THRESHOLD = 0.05  # 5% difference for trend detection

# Metric field sets per mode
_COMPACT_FIELDS = {"date", "sdnn", "rmssd", "lf_hf_ratio", "mean_hr"}
_MEDITATION_FIELDS = {
    "date",
    "hf_power",
    "lf_power",
    "lf_hf_ratio",
    "total_power",
    "sd1",
    "sd2",
    "sd1_sd2_ratio",
    "sample_entropy",
    "dfa_alpha1",
    "rsa",
    "mean_hr",
}
_FULL_FIELDS = {
    "date",
    "sdnn",
    "rmssd",
    "pnn50",
    "mean_nn",
    "hf_power",
    "lf_power",
    "lf_hf_ratio",
    "total_power",
    "sd1",
    "sd2",
    "sd1_sd2_ratio",
    "sample_entropy",
    "dfa_alpha1",
    "rsa",
    "mean_hr",
}

_MODE_FIELDS = {
    "compact": _COMPACT_FIELDS,
    "meditation": _MEDITATION_FIELDS,
    "full": _FULL_FIELDS,
}


# ===================================================================
# Public entry point
# ===================================================================


async def fetch_hrv_context_local(
    user_uid: str,
    hrv_range: str,
    mode: str = "compact",
) -> Dict[str, Any]:
    """
    Drop-in replacement for hrv_client.fetch_hrv_context.

    Returns:
        {
            "daily_14d": [...],
            "hrv_90d": {...}, "hr_90d": {...},
            "sleep_90d": {...}, "steps_90d": {...},
            "hrv_metrics": {...}
        }
    Returns {} on error or no data.
    """
    try:
        return await asyncio.to_thread(_compute_hrv_context, user_uid, hrv_range, mode)
    except Exception as exc:
        logger.warning("Local HRV computation failed: %s", exc)
        return {}


# ===================================================================
# Orchestrator
# ===================================================================


def _compute_hrv_context(
    user_uid: str, hrv_range: str, mode: str
) -> Dict[str, Any]:
    eng = get_engine()
    with eng.begin() as conn:
        if _APPLE_HEALTH_ONLY:
            daily_14d = _daily_from_apple_sdnn(conn, user_uid)
            hr_daily = _daily_heart_rate(conn, user_uid)
            for row in daily_14d:
                d = row["date"]
                row["mean_hr"] = round(hr_daily[d], 1) if d in hr_daily else None
            latest_metrics = None
        else:
            tier = _detect_tier(conn, user_uid)
            daily_14d = _query_daily_14d(conn, user_uid, tier)
            latest_metrics = _compute_latest_session_metrics(conn, user_uid, tier)
        aggregates = _compute_aggregates(conn, user_uid, _AGG_WINDOW)

    if not daily_14d and not aggregates:
        return {}

    # Filter daily records by mode
    fields = _MODE_FIELDS.get(mode, _COMPACT_FIELDS)
    filtered_daily = [_filter_by_mode(row, fields) for row in daily_14d]

    result: Dict[str, Any] = {}
    if filtered_daily:
        result["daily_14d"] = filtered_daily
    result.update(aggregates)
    if latest_metrics:
        result["hrv_metrics"] = _filter_by_mode(latest_metrics, fields)
    return result


# ===================================================================
# Tier detection
# ===================================================================


def _detect_tier(conn: Any, user_uid: str) -> int:
    """
    Determine the best available data tier for a user.
    Returns 1, 2, or 3.
    """
    row = conn.execute(
        text("""
            SELECT
                COUNT(*) FILTER (WHERE sample_type = 'heartbeat_series' AND payload IS NOT NULL) AS tier1,
                COUNT(*) FILTER (WHERE sample_type = 'hrv_sdnn' AND payload IS NOT NULL) AS tier2,
                COUNT(*) FILTER (WHERE sample_type = 'hrv') AS tier3
            FROM health_samples
            WHERE user_id = :uid
              AND start_time >= NOW() - INTERVAL '14 days'
        """),
        {"uid": user_uid},
    ).fetchone()

    if row.tier1 and row.tier1 > 0:
        return 1
    if row.tier2 and row.tier2 > 0:
        return 2
    return 3


# ===================================================================
# Daily 14-day metrics
# ===================================================================


def _query_daily_14d(
    conn: Any, user_uid: str, tier: int
) -> List[Dict[str, Any]]:
    """Build 14-day daily HRV records based on the available tier."""
    if tier == 1:
        hrv_daily = _daily_from_heartbeat_series(conn, user_uid)
    elif tier == 2:
        hrv_daily = _daily_from_hrv_sdnn_payload(conn, user_uid)
    else:
        hrv_daily = _daily_from_apple_sdnn(conn, user_uid)

    # Merge with daily heart rate averages
    hr_daily = _daily_heart_rate(conn, user_uid)
    for row in hrv_daily:
        d = row["date"]
        if d in hr_daily:
            row["mean_hr"] = round(hr_daily[d], 1)
        else:
            row["mean_hr"] = None

    return hrv_daily


# --- Tier 3: Apple SDNN only (current data) ---

def _daily_from_apple_sdnn(conn: Any, user_uid: str) -> List[Dict[str, Any]]:
    """Tier 3: aggregate Apple HRV SDNN values by day."""
    rows = conn.execute(
        text("""
            SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                   AVG(value) AS avg_sdnn,
                   COUNT(*) AS samples
            FROM health_samples
            WHERE user_id = :uid
              AND sample_type = 'hrv'
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
            GROUP BY DATE(start_time AT TIME ZONE 'UTC')
            ORDER BY day ASC
        """.replace(":days", str(_DAILY_WINDOW))),
        {"uid": user_uid},
    ).fetchall()

    daily = []
    for r in rows:
        daily.append({
            "date": str(r.day),
            "sdnn": round(r.avg_sdnn, 2),
            "rmssd": None,
            "pnn50": None,
            "mean_nn": None,
            "hf_power": None,
            "lf_power": None,
            "lf_hf_ratio": None,
            "total_power": None,
            "sd1": None,
            "sd2": None,
            "sd1_sd2_ratio": None,
            "sample_entropy": None,
            "dfa_alpha1": None,
            "rsa": None,
            "mean_hr": None,  # filled later
        })
    return daily[-_DAILY_WINDOW:]


# --- Tier 2: hrv_sdnn with beat_to_beat_bpm payload ---

def _daily_from_hrv_sdnn_payload(conn: Any, user_uid: str) -> List[Dict[str, Any]]:
    """Tier 2: extract BPM arrays from payload, convert to RR, compute via NeuroKit2."""
    rows = conn.execute(
        text("""
            SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                   payload,
                   value AS apple_sdnn
            FROM health_samples
            WHERE user_id = :uid
              AND sample_type = 'hrv_sdnn'
              AND payload IS NOT NULL
              AND start_time >= NOW() - INTERVAL ':days days'
            ORDER BY start_time ASC
        """.replace(":days", str(_DAILY_WINDOW))),
        {"uid": user_uid},
    ).fetchall()

    # Group by day, collect RR intervals
    day_rr: Dict[str, List[float]] = {}
    for r in rows:
        d = str(r.day)
        payload = r.payload if isinstance(r.payload, dict) else json.loads(r.payload)
        bpm_list = payload.get("beat_to_beat_bpm", [])
        rr_intervals = _bpm_list_to_rr(bpm_list)
        if rr_intervals:
            day_rr.setdefault(d, []).extend(rr_intervals)

    daily = []
    for d in sorted(day_rr.keys()):
        metrics = _compute_rr_metrics(day_rr[d])
        metrics["date"] = d
        metrics["mean_hr"] = None  # filled later
        daily.append(metrics)

    return daily[-_DAILY_WINDOW:]


# --- Tier 1: heartbeat_series with RR intervals ---

def _daily_from_heartbeat_series(conn: Any, user_uid: str) -> List[Dict[str, Any]]:
    """
    Tier 1: extract RR intervals from heartbeat_series payload.

    The ingest service pre-computes metrics and stores them in
    payload->'computed_metrics'. If RR intervals have been trimmed
    (only last 2 sessions keep raw RR), use the pre-computed metrics.
    If raw RR is still present, compute fresh via NeuroKit2.
    """
    rows = conn.execute(
        text("""
            SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                   payload
            FROM health_samples
            WHERE user_id = :uid
              AND sample_type = 'heartbeat_series'
              AND payload IS NOT NULL
              AND start_time >= NOW() - INTERVAL ':days days'
            ORDER BY start_time ASC
        """.replace(":days", str(_DAILY_WINDOW))),
        {"uid": user_uid},
    ).fetchall()

    # Collect per-day: either RR intervals (for fresh compute) or pre-computed metrics
    day_rr: Dict[str, List[float]] = {}
    day_precomputed: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        d = str(r.day)
        payload = r.payload if isinstance(r.payload, dict) else json.loads(r.payload)

        # Try raw RR intervals first (available for recent sessions)
        rr_list = payload.get("rr_intervals", [])
        rr_intervals = [
            entry["rr_interval_ms"]
            for entry in rr_list
            if isinstance(entry, dict) and "rr_interval_ms" in entry
        ]
        if rr_intervals:
            day_rr.setdefault(d, []).extend(rr_intervals)
        elif "computed_metrics" in payload:
            # RR was trimmed; use pre-computed metrics from ingest
            day_precomputed.setdefault(d, []).append(payload["computed_metrics"])

    daily = []

    # Days with raw RR → compute fresh
    for d in sorted(day_rr.keys()):
        metrics = _compute_rr_metrics(day_rr[d])
        metrics["date"] = d
        metrics["mean_hr"] = None  # filled later
        daily.append(metrics)

    # Days with only pre-computed metrics → average them
    for d in sorted(day_precomputed.keys()):
        if d in day_rr:
            continue  # already handled via raw RR
        sessions = day_precomputed[d]
        merged = _average_precomputed_metrics(sessions)
        merged["date"] = d
        merged["mean_hr"] = None  # filled later
        daily.append(merged)

    # Sort by date
    daily.sort(key=lambda x: x["date"])
    return daily[-_DAILY_WINDOW:]


# ===================================================================
# Daily heart rate
# ===================================================================


def _daily_heart_rate(conn: Any, user_uid: str) -> Dict[str, float]:
    """Return {date_str: mean_hr} for last 14 days."""
    rows = conn.execute(
        text("""
            SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                   AVG(value) AS mean_hr
            FROM health_samples
            WHERE user_id = :uid
              AND sample_type = 'heart_rate'
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
            GROUP BY DATE(start_time AT TIME ZONE 'UTC')
        """.replace(":days", str(_DAILY_WINDOW))),
        {"uid": user_uid},
    ).fetchall()
    return {str(r.day): float(r.mean_hr) for r in rows}


# ===================================================================
# NeuroKit2 HRV computation from RR intervals
# ===================================================================


def _compute_rr_metrics(rr_intervals: List[float]) -> Dict[str, Any]:
    """
    Compute all HRV metrics from RR intervals (in milliseconds).

    Uses nk.hrv() when NeuroKit2 is available; falls back to numpy basics.
    Returns a dict with all metric fields (nulls where not computable).
    """
    empty = {
        "sdnn": None, "rmssd": None, "pnn50": None, "mean_nn": None,
        "hf_power": None, "lf_power": None, "lf_hf_ratio": None, "total_power": None,
        "sd1": None, "sd2": None, "sd1_sd2_ratio": None,
        "sample_entropy": None, "dfa_alpha1": None, "rsa": None,
    }

    if not rr_intervals or len(rr_intervals) < 10:
        return empty

    if _HAS_NEUROKIT:
        return _compute_rr_neurokit(rr_intervals)

    # Fallback: numpy-only time-domain basics
    if np is None:
        return empty
    return _compute_rr_numpy(rr_intervals)


def _compute_rr_neurokit(rr_intervals: List[float]) -> Dict[str, Any]:
    """Full HRV via NeuroKit2's nk.hrv() — time, frequency, nonlinear."""
    rr = np.array(rr_intervals, dtype=np.float64)

    # NeuroKit2 expects R-peak indices at a given sampling rate.
    # Convert RR intervals (ms) to cumulative peak positions at 1000 Hz.
    peaks = np.cumsum(rr) / 1000.0 * 1000  # positions in samples at 1000 Hz
    peaks = np.insert(peaks, 0, 0).astype(int)

    result: Dict[str, Any] = {
        "sdnn": None, "rmssd": None, "pnn50": None, "mean_nn": None,
        "hf_power": None, "lf_power": None, "lf_hf_ratio": None, "total_power": None,
        "sd1": None, "sd2": None, "sd1_sd2_ratio": None,
        "sample_entropy": None, "dfa_alpha1": None, "rsa": None,
    }

    try:
        hrv_all = nk.hrv(peaks, sampling_rate=1000, show=False)

        # Time-domain
        result["sdnn"] = _safe_round(hrv_all, "HRV_SDNN")
        result["rmssd"] = _safe_round(hrv_all, "HRV_RMSSD")
        result["pnn50"] = _safe_round(hrv_all, "HRV_pNN50")
        result["mean_nn"] = _safe_round(hrv_all, "HRV_MeanNN")

        # Frequency-domain
        result["hf_power"] = _safe_round(hrv_all, "HRV_HF")
        result["lf_power"] = _safe_round(hrv_all, "HRV_LF")
        result["lf_hf_ratio"] = _safe_round(hrv_all, "HRV_LFHF")
        result["total_power"] = _safe_round(hrv_all, "HRV_TP")

        # Nonlinear
        result["sd1"] = _safe_round(hrv_all, "HRV_SD1")
        result["sd2"] = _safe_round(hrv_all, "HRV_SD2")
        result["sd1_sd2_ratio"] = _safe_round(hrv_all, "HRV_SD1SD2")
        result["sample_entropy"] = _safe_round(hrv_all, "HRV_SampEn")
        result["dfa_alpha1"] = _safe_round(hrv_all, "HRV_DFA_alpha1")

        # RSA (approximated from HF power — true RSA needs respiration signal)
        result["rsa"] = _safe_round(hrv_all, "HRV_HF")

    except Exception as exc:
        logger.warning("NeuroKit2 HRV computation error: %s", exc)
        # Fall back to numpy basics
        return _compute_rr_numpy(rr_intervals)

    return result


def _compute_rr_numpy(rr_intervals: List[float]) -> Dict[str, Any]:
    """Time-domain HRV from numpy only (no frequency/nonlinear)."""
    rr = np.array(rr_intervals, dtype=np.float64)
    diffs = np.diff(rr)

    result: Dict[str, Any] = {
        "sdnn": round(float(np.std(rr, ddof=1)), 2),
        "rmssd": round(float(np.sqrt(np.mean(diffs ** 2))), 2),
        "pnn50": round(float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100), 2),
        "mean_nn": round(float(np.mean(rr)), 2),
        "hf_power": None,
        "lf_power": None,
        "lf_hf_ratio": None,
        "total_power": None,
        "sd1": None,
        "sd2": None,
        "sd1_sd2_ratio": None,
        "sample_entropy": None,
        "dfa_alpha1": None,
        "rsa": None,
        "mean_hr": None,
    }

    # Poincaré SD1/SD2 can be computed from numpy
    sd1 = float(np.std(diffs / np.sqrt(2), ddof=1))
    sd2 = float(np.std(rr[:-1] + rr[1:], ddof=1) / np.sqrt(2))
    result["sd1"] = round(sd1, 2)
    result["sd2"] = round(sd2, 2)
    if sd2 > 0:
        result["sd1_sd2_ratio"] = round(sd1 / sd2, 3)

    return result


def _safe_round(df: Any, col: str, decimals: int = 2) -> Optional[float]:
    """Safely extract a value from a NeuroKit2 result DataFrame."""
    try:
        val = df[col].iloc[0]
        if val is None or (isinstance(val, float) and (val != val)):  # NaN check
            return None
        return round(float(val), decimals)
    except (KeyError, IndexError, TypeError):
        return None


# ===================================================================
# BPM → RR conversion
# ===================================================================


def _average_precomputed_metrics(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average pre-computed HRV metrics from multiple sessions in a day."""
    all_keys = {
        "sdnn", "rmssd", "pnn50", "mean_nn",
        "hf_power", "lf_power", "lf_hf_ratio", "total_power",
        "sd1", "sd2", "sd1_sd2_ratio", "sample_entropy", "dfa_alpha1", "rsa",
    }
    result: Dict[str, Any] = {}
    for key in all_keys:
        vals = [s[key] for s in sessions if key in s and s[key] is not None]
        if vals:
            result[key] = round(sum(vals) / len(vals), 2)
        else:
            result[key] = None
    return result


def _bpm_list_to_rr(bpm_list: Any) -> List[float]:
    """Convert beat-to-beat BPM array to RR intervals in ms."""
    if not isinstance(bpm_list, list):
        return []
    rr = []
    for entry in bpm_list:
        if isinstance(entry, dict):
            bpm = entry.get("bpm")
        elif isinstance(entry, (int, float)):
            bpm = entry
        else:
            continue
        if bpm and bpm > 20 and bpm < 250:
            rr.append(60000.0 / bpm)
    return rr


# ===================================================================
# 90-day aggregates
# ===================================================================


def _compute_aggregates(
    conn: Any, user_uid: str, days: int = 90
) -> Dict[str, Any]:
    """Compute aggregate health context blocks for the prompt."""
    result: Dict[str, Any] = {}

    hrv_agg = _agg_hrv(conn, user_uid, days)
    if hrv_agg:
        result["hrv_90d"] = hrv_agg

    hrv_sdnn_agg = _agg_hrv_sdnn(conn, user_uid, days)
    if hrv_sdnn_agg:
        result["hrv_sdnn_90d"] = hrv_sdnn_agg

    hr_agg = _agg_heart_rate(conn, user_uid, days)
    if hr_agg:
        result["hr_90d"] = hr_agg

    sleep_agg = _agg_sleep(conn, user_uid, days)
    if sleep_agg:
        result["sleep_90d"] = sleep_agg

    steps_agg = _agg_steps(conn, user_uid, days)
    if steps_agg:
        result["steps_90d"] = steps_agg

    return result


def _agg_hrv(conn: Any, user_uid: str, days: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        text("""
            SELECT AVG(value) AS mean_sdnn, COUNT(*) AS cnt
            FROM health_samples
            WHERE user_id = :uid AND sample_type = 'hrv'
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
        """.replace(":days", str(days))),
        {"uid": user_uid},
    ).fetchone()

    if not row or not row.cnt or row.cnt == 0:
        return None

    trend = _half_split_trend(conn, user_uid, "hrv", days)
    return {"mean_sdnn": round(float(row.mean_sdnn), 2), "trend": trend}


def _agg_hrv_sdnn(conn: Any, user_uid: str, days: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        text("""
            SELECT
                AVG(value) AS mean_sdnn,
                COUNT(*) AS session_count,
                COUNT(*) FILTER (WHERE payload IS NOT NULL) AS payload_count,
                COUNT(*) FILTER (WHERE value IS NOT NULL) AS value_count
            FROM health_samples
            WHERE user_id = :uid AND sample_type = 'hrv_sdnn'
              AND start_time >= NOW() - INTERVAL ':days days'
        """.replace(":days", str(days))),
        {"uid": user_uid},
    ).fetchone()

    if not row or not row.session_count or row.session_count == 0:
        return None

    result: Dict[str, Any] = {
        "session_count": int(row.session_count),
        "payload_count": int(row.payload_count or 0),
        "value_count": int(row.value_count or 0),
    }
    if row.mean_sdnn is not None:
        result["mean_sdnn"] = round(float(row.mean_sdnn), 2)
        result["trend"] = _half_split_trend(conn, user_uid, "hrv_sdnn", days)
    return result


def _agg_heart_rate(conn: Any, user_uid: str, days: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        text("""
            SELECT AVG(value) AS mean,
                   PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY value) AS p10,
                   PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY value) AS p90,
                   COUNT(*) AS cnt
            FROM health_samples
            WHERE user_id = :uid AND sample_type = 'heart_rate'
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
        """.replace(":days", str(days))),
        {"uid": user_uid},
    ).fetchone()

    if not row or not row.cnt or row.cnt == 0:
        return None

    return {
        "mean": round(float(row.mean), 1),
        "p10": round(float(row.p10), 1),
        "p90": round(float(row.p90), 1),
    }


def _agg_sleep(conn: Any, user_uid: str, days: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        text("""
            SELECT AVG(value) AS mean_hours, COUNT(*) AS cnt
            FROM health_samples
            WHERE user_id = :uid AND sample_type = 'sleep'
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
        """.replace(":days", str(days))),
        {"uid": user_uid},
    ).fetchone()

    if not row or not row.cnt or row.cnt == 0:
        return None

    trend = _half_split_trend(conn, user_uid, "sleep", days)
    return {"mean_hours": round(float(row.mean_hours), 1), "trend": trend}


def _agg_steps(conn: Any, user_uid: str, days: int) -> Optional[Dict[str, Any]]:
    """Daily average step count over the window."""
    row = conn.execute(
        text("""
            SELECT AVG(daily_total) AS mean, COUNT(*) AS cnt
            FROM (
                SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                       SUM(value) AS daily_total
                FROM health_samples
                WHERE user_id = :uid AND sample_type = 'steps'
                  AND start_time >= NOW() - INTERVAL ':days days'
                  AND value IS NOT NULL
                GROUP BY DATE(start_time AT TIME ZONE 'UTC')
            ) AS daily
        """.replace(":days", str(days))),
        {"uid": user_uid},
    ).fetchone()

    if not row or not row.cnt or row.cnt == 0:
        return None

    trend = _half_split_trend(conn, user_uid, "steps", days)
    return {"mean": round(float(row.mean), 0), "trend": trend}


# ===================================================================
# Trend computation (half-split)
# ===================================================================


def _half_split_trend(
    conn: Any, user_uid: str, sample_type: str, days: int
) -> str:
    """
    Compare recent half vs older half of the window.
    Returns "improving", "declining", or "stable".
    For HRV/sleep: higher = improving. For steps: higher = improving.
    """
    half = days // 2
    row = conn.execute(
        text("""
            SELECT
                AVG(value) FILTER (WHERE start_time >= NOW() - INTERVAL ':half days') AS recent,
                AVG(value) FILTER (WHERE start_time < NOW() - INTERVAL ':half days'
                                     AND start_time >= NOW() - INTERVAL ':days days') AS older
            FROM health_samples
            WHERE user_id = :uid AND sample_type = :stype
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
        """.replace(":half", str(half)).replace(":days", str(days))),
        {"uid": user_uid, "stype": sample_type},
    ).fetchone()

    if not row or row.recent is None or row.older is None or row.older == 0:
        return "stable"

    ratio = (float(row.recent) - float(row.older)) / abs(float(row.older))
    if ratio > _TREND_THRESHOLD:
        return "improving"
    elif ratio < -_TREND_THRESHOLD:
        return "declining"
    return "stable"


# ===================================================================
# Latest session metrics
# ===================================================================


def _compute_latest_session_metrics(
    conn: Any, user_uid: str, tier: int
) -> Optional[Dict[str, Any]]:
    """Compute HRV metrics from the most recent session (for hrv_metrics key)."""
    if tier == 1:
        row = conn.execute(
            text("""
                SELECT payload FROM health_samples
                WHERE user_id = :uid AND sample_type = 'heartbeat_series'
                  AND payload IS NOT NULL
                ORDER BY start_time DESC LIMIT 1
            """),
            {"uid": user_uid},
        ).fetchone()
        if row and row.payload:
            payload = row.payload if isinstance(row.payload, dict) else json.loads(row.payload)
            # Try raw RR first
            rr_list = payload.get("rr_intervals", [])
            rr = [e["rr_interval_ms"] for e in rr_list if isinstance(e, dict) and "rr_interval_ms" in e]
            if len(rr) >= 10:
                return _compute_rr_metrics(rr)
            # Fall back to pre-computed metrics (RR was trimmed)
            if "computed_metrics" in payload:
                return payload["computed_metrics"]

    elif tier == 2:
        row = conn.execute(
            text("""
                SELECT payload, value AS apple_sdnn FROM health_samples
                WHERE user_id = :uid AND sample_type = 'hrv_sdnn'
                  AND payload IS NOT NULL
                ORDER BY start_time DESC LIMIT 1
            """),
            {"uid": user_uid},
        ).fetchone()
        if row and row.payload:
            payload = row.payload if isinstance(row.payload, dict) else json.loads(row.payload)
            bpm_list = payload.get("beat_to_beat_bpm", [])
            rr = _bpm_list_to_rr(bpm_list)
            if len(rr) >= 10:
                return _compute_rr_metrics(rr)

    # Tier 3: no session-level metrics beyond SDNN
    return None


# ===================================================================
# Mode filtering
# ===================================================================


def _filter_by_mode(record: Dict[str, Any], fields: set) -> Dict[str, Any]:
    """Strip a record down to only the fields for the selected mode."""
    return {k: v for k, v in record.items() if k in fields and v is not None}
