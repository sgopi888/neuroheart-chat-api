"""
Mindfulness session tracking — records watch sessions with pre/post HRV comparison.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import psycopg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/mindfulness", tags=["mindfulness"])


# --- Schemas --- # redo

class RRInterval(BaseModel):
    rr_interval_ms: float
    timestamp: Optional[float] = None

class SessionIn(BaseModel):
    user_id: str = Field(min_length=8, max_length=128)
    start_time: str
    end_time: str
    duration_minutes: int
    mood: Optional[str] = None
    depth: Optional[str] = None
    source: str = "watch"
    beginning_rr: List[RRInterval] = []
    ending_rr: List[RRInterval] = []


from app.hrv_utils import compute_hrv_from_rr as _compute_hrv_from_rr
from app.hrv_bpm_per_min import process_bpm_session, save_session_results, load_cross_session_baseline

_CALM_LINK_WINDOW_S = 120  # seconds tolerance for matching calm_score_session


def _try_link_calm(cur, user_id: str, start_time: str) -> tuple:
    """Try to find a matching calm_score_session and return (ref_id, summary_dict)."""
    cur.execute(
        """
        SELECT id, payload->'summary' AS summary
        FROM health_samples
        WHERE user_id = %s
          AND sample_type = 'calm_score_session'
          AND ABS(EXTRACT(EPOCH FROM start_time - %s::timestamptz)) < %s
        ORDER BY ABS(EXTRACT(EPOCH FROM start_time - %s::timestamptz))
        LIMIT 1
        """,
        (user_id, start_time, _CALM_LINK_WINDOW_S, start_time),
    )
    row = cur.fetchone()
    if row:
        ref_id = row[0]
        summary = row[1] if isinstance(row[1], dict) else (json.loads(row[1]) if row[1] else None)
        return ref_id, summary
    return None, None


def _compute_delta(beginning: Dict, ending: Dict) -> Dict[str, Any]:
    """Compute change between beginning and ending HRV metrics."""
    delta = {}
    for key in ["sdnn", "rmssd", "pnn50", "mean_hr"]:
        b = beginning.get(key)
        e = ending.get(key)
        if b is not None and e is not None and b > 0:
            delta[key] = round(e - b, 2)
            delta[f"{key}_pct"] = round(((e - b) / b) * 100, 1)
    # Positive SDNN/RMSSD delta = improvement (more relaxed)
    sdnn_d = delta.get("sdnn", 0)
    rmssd_d = delta.get("rmssd", 0)
    if sdnn_d > 2 or rmssd_d > 3:
        delta["outcome"] = "improved"
    elif sdnn_d < -2 or rmssd_d < -3:
        delta["outcome"] = "declined"
    else:
        delta["outcome"] = "stable"
    return delta


# --- Endpoints ---

@router.post("/session")
async def record_session(body: SessionIn):
    """Record a completed mindfulness session with pre/post HRV data."""

    beginning_rr_vals = [r.rr_interval_ms for r in body.beginning_rr]
    ending_rr_vals = [r.rr_interval_ms for r in body.ending_rr]

    beginning_hrv = _compute_hrv_from_rr(beginning_rr_vals)
    ending_hrv = _compute_hrv_from_rr(ending_rr_vals)

    hrv_delta = None
    if beginning_hrv and ending_hrv:
        hrv_delta = _compute_delta(beginning_hrv, ending_hrv)

    # Full-session HRV from all RR intervals combined
    all_rr_vals = beginning_rr_vals + ending_rr_vals
    session_hrv = _compute_hrv_from_rr(all_rr_vals)

    # Run calm-score pipeline inline on the RR intervals
    calm_ref = None
    calm_summary = None
    rr_samples = [{"rr_interval_ms": v} for v in all_rr_vals]
    if len(rr_samples) >= 30:
        try:
            cross_bl = load_cross_session_baseline(body.user_id)
            snapshots, summary = process_bpm_session(rr_samples, cross_bl)
            if snapshots:
                saved = save_session_results(body.user_id, body.start_time, snapshots, summary)
                logger.info(
                    "Mindfulness calm pipeline: %d snapshots, avg_calm=%.1f, saved=%d",
                    len(snapshots), summary.avg_calm_score, saved,
                )
                from dataclasses import asdict
                calm_summary = asdict(summary)
                for k, v in calm_summary.items():
                    if isinstance(v, float):
                        calm_summary[k] = round(v, 2)
        except Exception:
            logger.exception("Mindfulness calm pipeline failed for user=%s", body.user_id[:12])

    with psycopg.connect(settings.database_url_psycopg) as conn:
        with conn.cursor() as cur:
            # If we didn't compute calm inline, try to link to an existing calm_score_session
            if not calm_summary:
                calm_ref, calm_summary = _try_link_calm(cur, body.user_id, body.start_time)

            cur.execute(
                """
                INSERT INTO mindfulness_sessions
                    (user_id, start_time, end_time, duration_minutes, mood, depth, source,
                     beginning_hrv, ending_hrv, hrv_delta,
                     session_hrv, calm_score_ref, calm_summary)
                VALUES (%s, %s::timestamptz, %s::timestamptz, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    body.user_id,
                    body.start_time,
                    body.end_time,
                    body.duration_minutes,
                    body.mood,
                    body.depth,
                    body.source,
                    json.dumps(beginning_hrv) if beginning_hrv else None,
                    json.dumps(ending_hrv) if ending_hrv else None,
                    json.dumps(hrv_delta) if hrv_delta else None,
                    json.dumps(session_hrv) if session_hrv else None,
                    calm_ref,
                    json.dumps(calm_summary) if calm_summary else None,
                ),
            )
            session_id = cur.fetchone()[0]

    return {
        "session_id": session_id,
        "beginning_hrv": beginning_hrv,
        "ending_hrv": ending_hrv,
        "hrv_delta": hrv_delta,
        "session_hrv": session_hrv,
        "calm_summary": calm_summary,
    }


@router.get("/sessions")
async def list_sessions(
    user_id: str = Query(..., min_length=8),
    limit: int = Query(default=20, le=100),
):
    """List past mindfulness sessions with HRV results."""
    with psycopg.connect(settings.database_url_psycopg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, start_time, end_time, duration_minutes, mood, depth, source,
                       beginning_hrv, ending_hrv, hrv_delta, created_at,
                       session_hrv, calm_score_ref, calm_summary
                FROM mindfulness_sessions
                WHERE user_id = %s
                ORDER BY start_time DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            rows = cur.fetchall()

    sessions = []
    for r in rows:
        sessions.append({
            "id": r[0],
            "start_time": r[1].isoformat(),
            "end_time": r[2].isoformat(),
            "duration_minutes": r[3],
            "mood": r[4],
            "depth": r[5],
            "source": r[6],
            "beginning_hrv": r[7],
            "ending_hrv": r[8],
            "hrv_delta": r[9],
            "created_at": r[10].isoformat(),
            "session_hrv": r[11],
            "calm_score_ref": r[12],
            "calm_summary": r[13],
        })

    return {"user_id": user_id, "sessions": sessions}


@router.get("/session/{session_id}")
async def get_session(session_id: int):
    """Get a single mindfulness session with full HRV detail and calm score snapshots."""
    with psycopg.connect(settings.database_url_psycopg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ms.id, ms.user_id, ms.start_time, ms.end_time,
                       ms.duration_minutes, ms.mood, ms.depth, ms.source,
                       ms.beginning_hrv, ms.ending_hrv, ms.hrv_delta,
                       ms.session_hrv, ms.calm_score_ref, ms.calm_summary,
                       ms.created_at,
                       hs.payload AS calm_payload
                FROM mindfulness_sessions ms
                LEFT JOIN health_samples hs
                  ON hs.id = ms.calm_score_ref
                WHERE ms.id = %s
                """,
                (session_id,),
            )
            r = cur.fetchone()

    if not r:
        raise HTTPException(status_code=404, detail="Session not found")

    result = {
        "id": r[0],
        "user_id": r[1],
        "start_time": r[2].isoformat(),
        "end_time": r[3].isoformat(),
        "duration_minutes": r[4],
        "mood": r[5],
        "depth": r[6],
        "source": r[7],
        "beginning_hrv": r[8],
        "ending_hrv": r[9],
        "hrv_delta": r[10],
        "session_hrv": r[11],
        "calm_score_ref": r[12],
        "calm_summary": r[13],
        "created_at": r[14].isoformat(),
    }

    # Include calm score snapshots timeseries if linked
    calm_payload = r[15]
    if calm_payload:
        if isinstance(calm_payload, str):
            calm_payload = json.loads(calm_payload)
        result["calm_snapshots"] = calm_payload.get("snapshots")
    else:
        result["calm_snapshots"] = None

    return result
