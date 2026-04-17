from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.db import get_engine

logger = logging.getLogger(__name__)

_DAILY_WINDOW = 14
_TIMESERIES_WINDOW = 14
_DAILY_HOURLY_WINDOW = 30
_AGG_WINDOW = 90
_TREND_THRESHOLD = 0.05


async def fetch_hrv_context_apple(
    user_uid: str,
    hrv_range: str,
    mode: str = "compact",
) -> Dict[str, Any]:
    """Return Apple Health HRV context only; no NeuroKit computation."""
    del hrv_range, mode
    try:
        return await asyncio.to_thread(_compute_hrv_context_apple, user_uid)
    except Exception as exc:
        logger.warning("Apple HRV context fetch failed: %s", exc)
        return {}


def _compute_hrv_context_apple(user_uid: str) -> Dict[str, Any]:
    eng = get_engine()
    with eng.begin() as conn:
        daily_14d = _daily_from_apple_sdnn(conn, user_uid)
        hr_daily = _daily_heart_rate(conn, user_uid)
        for row in daily_14d:
            d = row["date"]
            row["mean_hr"] = round(hr_daily[d], 1) if d in hr_daily else None

        daily_90d = _daily_from_apple_sdnn_range(conn, user_uid, _AGG_WINDOW)
        hr_daily_90 = _daily_heart_rate_range(conn, user_uid, _AGG_WINDOW)
        for row in daily_90d:
            d = row["date"]
            row["mean_hr"] = round(hr_daily_90[d], 1) if d in hr_daily_90 else None

        hrv_daily_hourly_30d = _daily_hourly_timeseries(conn, user_uid, "hrv", _DAILY_HOURLY_WINDOW)
        hrv_sdnn_daily_hourly_30d = _daily_hourly_timeseries(
            conn, user_uid, "hrv_sdnn", _DAILY_HOURLY_WINDOW
        )
        aggregates = _compute_aggregates(conn, user_uid, _AGG_WINDOW)

        calm_sessions = _recent_calm_sessions(conn, user_uid)
        mindfulness_sessions = _recent_mindfulness_sessions(conn, user_uid)

    if (
        not daily_14d
        and not daily_90d
        and not hrv_daily_hourly_30d
        and not hrv_sdnn_daily_hourly_30d
        and not aggregates
        and not calm_sessions
        and not mindfulness_sessions
    ):
        return {}

    result: Dict[str, Any] = {}
    if daily_14d:
        result["daily_14d"] = daily_14d
    if daily_90d:
        result["daily_90d"] = daily_90d
    if hrv_daily_hourly_30d:
        result["hrv_daily_hourly_30d"] = hrv_daily_hourly_30d
    if hrv_sdnn_daily_hourly_30d:
        result["hrv_sdnn_daily_hourly_30d"] = hrv_sdnn_daily_hourly_30d
    if calm_sessions:
        result["calm_score_sessions"] = calm_sessions
    if mindfulness_sessions:
        result["mindfulness_sessions"] = mindfulness_sessions
    result.update(aggregates)
    return result


def _daily_from_apple_sdnn(conn: Any, user_uid: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text("""
            SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                   AVG(value) AS avg_sdnn
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
    for row in rows:
        daily.append({
            "date": str(row.day),
            "sdnn": round(float(row.avg_sdnn), 2),
            "mean_hr": None,
        })
    return daily[-_DAILY_WINDOW:]


def _daily_from_apple_sdnn_range(conn: Any, user_uid: str, days: int) -> List[Dict[str, Any]]:
    """Daily SDNN averages for arbitrary day range."""
    rows = conn.execute(
        text("""
            SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                   AVG(value) AS avg_sdnn
            FROM health_samples
            WHERE user_id = :uid
              AND sample_type = 'hrv'
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
            GROUP BY DATE(start_time AT TIME ZONE 'UTC')
            ORDER BY day ASC
        """.replace(":days", str(days))),
        {"uid": user_uid},
    ).fetchall()

    return [
        {"date": str(row.day), "sdnn": round(float(row.avg_sdnn), 2), "mean_hr": None}
        for row in rows
    ]


def _daily_heart_rate_range(conn: Any, user_uid: str, days: int) -> Dict[str, float]:
    """Daily heart rate averages for arbitrary day range."""
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
        """.replace(":days", str(days))),
        {"uid": user_uid},
    ).fetchall()
    return {str(row.day): float(row.mean_hr) for row in rows}


def _daily_heart_rate(conn: Any, user_uid: str) -> Dict[str, float]:
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
    return {str(row.day): float(row.mean_hr) for row in rows}


def _sample_timeseries(
    conn: Any, user_uid: str, sample_type: str, days: int
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text("""
            SELECT start_time AT TIME ZONE 'UTC' AS ts,
                   value
            FROM health_samples
            WHERE user_id = :uid
              AND sample_type = :sample_type
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
            ORDER BY start_time ASC
        """.replace(":days", str(days))),
        {"uid": user_uid, "sample_type": sample_type},
    ).fetchall()

    return [
        {"timestamp": row.ts.isoformat(), "value": round(float(row.value), 2)}
        for row in rows
    ]


def _daily_hourly_timeseries(
    conn: Any, user_uid: str, sample_type: str, days: int
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text("""
            SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                   FLOOR(EXTRACT(HOUR FROM start_time AT TIME ZONE 'UTC') / 2) * 2 AS hour_bucket,
                   AVG(value) AS avg_value,
                   COUNT(*) AS samples
            FROM health_samples
            WHERE user_id = :uid
              AND sample_type = :sample_type
              AND start_time >= NOW() - INTERVAL ':days days'
              AND value IS NOT NULL
            GROUP BY DATE(start_time AT TIME ZONE 'UTC'),
                     FLOOR(EXTRACT(HOUR FROM start_time AT TIME ZONE 'UTC') / 2) * 2
            ORDER BY day ASC, hour_bucket ASC
        """.replace(":days", str(days))),
        {"uid": user_uid, "sample_type": sample_type},
    ).fetchall()

    result = []
    for row in rows:
        hour = int(row.hour_bucket)
        result.append({
            "date": str(row.day),
            "window": f"{hour:02d}:00-{hour + 2:02d}:00",
            "avg_value": round(float(row.avg_value), 2),
            "samples": int(row.samples),
        })
    return result


def _compute_aggregates(conn: Any, user_uid: str, days: int = 90) -> Dict[str, Any]:
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

    if not row or not row.cnt:
        return None

    return {
        "mean_sdnn": round(float(row.mean_sdnn), 2),
        "trend": _half_split_trend(conn, user_uid, "hrv", days),
    }


def _agg_hrv_sdnn(conn: Any, user_uid: str, days: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        text("""
            SELECT AVG(value) AS mean_sdnn,
                   COUNT(*) AS session_count,
                   COUNT(*) FILTER (WHERE payload IS NOT NULL) AS payload_count,
                   COUNT(*) FILTER (WHERE value IS NOT NULL) AS value_count
            FROM health_samples
            WHERE user_id = :uid AND sample_type = 'hrv_sdnn'
              AND start_time >= NOW() - INTERVAL ':days days'
        """.replace(":days", str(days))),
        {"uid": user_uid},
    ).fetchone()

    if not row or not row.session_count:
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

    if not row or not row.cnt:
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

    if not row or not row.cnt:
        return None

    return {
        "mean_hours": round(float(row.mean_hours), 1),
        "trend": _half_split_trend(conn, user_uid, "sleep", days),
    }


def _agg_steps(conn: Any, user_uid: str, days: int) -> Optional[Dict[str, Any]]:
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

    if not row or not row.cnt:
        return None

    return {
        "mean": round(float(row.mean), 0),
        "trend": _half_split_trend(conn, user_uid, "steps", days),
    }


def _recent_calm_sessions(conn: Any, user_uid: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return the most recent calm_score session summaries for LLM context."""
    rows = conn.execute(
        text("""
            SELECT start_time AT TIME ZONE 'UTC' AS ts,
                   value AS avg_calm_score,
                   payload->'summary' AS summary
            FROM health_samples
            WHERE user_id = :uid
              AND sample_type = 'calm_score_session'
              AND value IS NOT NULL
            ORDER BY start_time DESC
            LIMIT :lim
        """),
        {"uid": user_uid, "lim": limit},
    ).fetchall()

    sessions = []
    for row in rows:
        entry: Dict[str, Any] = {
            "date": row.ts.isoformat() if row.ts else None,
            "avg_calm_score": round(float(row.avg_calm_score), 1),
        }
        if row.summary and isinstance(row.summary, dict):
            s = row.summary
            entry["hr_baseline"] = s.get("hr_baseline")
            entry["hr_final"] = s.get("hr_final")
            entry["hr_delta"] = s.get("hr_delta")
            entry["hf_pct_change"] = s.get("hf_pct_change")
            entry["breath_start"] = s.get("breath_start")
            entry["breath_end"] = s.get("breath_end")
            entry["duration_s"] = s.get("duration_s")
            entry["time_in_recovery_pct"] = s.get("time_in_recovery_pct")
            entry["time_in_stress_pct"] = s.get("time_in_stress_pct")
        sessions.append(entry)

    sessions.reverse()  # chronological order
    return sessions


def _recent_mindfulness_sessions(conn: Any, user_uid: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Return recent mindfulness sessions with per-track HRV for LLM context."""
    rows = conn.execute(
        text("""
            SELECT start_time AT TIME ZONE 'UTC' AS ts,
                   duration_minutes,
                   mood,
                   depth,
                   session_hrv,
                   beginning_hrv,
                   ending_hrv,
                   hrv_delta,
                   calm_summary
            FROM mindfulness_sessions
            WHERE user_id = :uid
            ORDER BY start_time DESC
            LIMIT :lim
        """),
        {"uid": user_uid, "lim": limit},
    ).fetchall()

    sessions = []
    for row in rows:
        entry: Dict[str, Any] = {
            "date": row.ts.isoformat() if row.ts else None,
            "duration_minutes": row.duration_minutes,
            "mood": row.mood,
            "depth": row.depth,
        }
        if row.session_hrv and isinstance(row.session_hrv, dict):
            h = row.session_hrv
            entry["sdnn"] = h.get("sdnn")
            entry["rmssd"] = h.get("rmssd")
            entry["pnn50"] = h.get("pnn50")
            entry["mean_hr"] = h.get("mean_hr")
        if row.hrv_delta and isinstance(row.hrv_delta, dict):
            entry["delta_sdnn"] = row.hrv_delta.get("sdnn")
            entry["delta_rmssd"] = row.hrv_delta.get("rmssd")
            entry["outcome"] = row.hrv_delta.get("outcome")
        if row.calm_summary and isinstance(row.calm_summary, dict):
            entry["avg_calm_score"] = row.calm_summary.get("avg_calm_score")
            entry["time_in_recovery_pct"] = row.calm_summary.get("time_in_recovery_pct")
            entry["time_in_stress_pct"] = row.calm_summary.get("time_in_stress_pct")
        sessions.append(entry)

    sessions.reverse()  # chronological order
    return sessions


def _half_split_trend(conn: Any, user_uid: str, sample_type: str, days: int) -> str:
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
    if ratio < -_TREND_THRESHOLD:
        return "declining"
    return "stable"
