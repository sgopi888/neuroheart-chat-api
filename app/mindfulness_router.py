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

    with psycopg.connect(settings.database_url_psycopg) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mindfulness_sessions
                    (user_id, start_time, end_time, duration_minutes, mood, depth, source,
                     beginning_hrv, ending_hrv, hrv_delta)
                VALUES (%s, %s::timestamptz, %s::timestamptz, %s, %s, %s, %s, %s, %s, %s)
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
                ),
            )
            session_id = cur.fetchone()[0]

    return {
        "session_id": session_id,
        "beginning_hrv": beginning_hrv,
        "ending_hrv": ending_hrv,
        "hrv_delta": hrv_delta,
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
                       beginning_hrv, ending_hrv, hrv_delta, created_at
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
        })

    return {"user_id": user_id, "sessions": sessions}
