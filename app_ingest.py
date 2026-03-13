"""
Hardened health data ingest service.

Replaces /opt/neuroheart/app.py on the VPS.
Adds: batch size limits, sample type validation, per-user storage quotas,
      heartbeat_series → compute metrics + trim raw RR payload.

Run: uvicorn app_ingest:app --host 0.0.0.0 --port 8001
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import psycopg
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

app = FastAPI(title="NeuroHeart Ingest API", version="2.0")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_SAMPLES_PER_REQUEST = 5000
MAX_HEARTBEAT_SESSIONS = 2  # keep only last N raw heartbeat_series per user

ALLOWED_SAMPLE_TYPES = {
    "heart_rate",
    "hrv",
    "steps",
    "sleep",
    "hrv_sdnn",
    "heartbeat_series",
}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RegisterIn(BaseModel):
    user_id: str = Field(min_length=8, max_length=128)


class SampleIn(BaseModel):
    sample_type: str
    start_time: str
    end_time: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None


class IngestIn(BaseModel):
    user_id: str = Field(min_length=8, max_length=128)
    samples: List[SampleIn]


# ---------------------------------------------------------------------------
# HRV metrics computation (inline, no NeuroKit2 dependency in ingest)
# ---------------------------------------------------------------------------


def _compute_hrv_from_rr(rr_intervals: List[float]) -> Optional[Dict[str, Any]]:
    """
    Compute basic HRV metrics from RR intervals (ms).
    Returns a compact metrics dict, or None if insufficient data.
    Uses only stdlib + math (no numpy/neurokit2 in ingest service).
    """
    if len(rr_intervals) < 10:
        return None

    n = len(rr_intervals)
    mean_nn = sum(rr_intervals) / n

    # SDNN
    variance = sum((x - mean_nn) ** 2 for x in rr_intervals) / (n - 1)
    sdnn = math.sqrt(variance)

    # RMSSD
    diffs = [rr_intervals[i + 1] - rr_intervals[i] for i in range(n - 1)]
    rmssd = math.sqrt(sum(d ** 2 for d in diffs) / len(diffs))

    # pNN50
    nn50 = sum(1 for d in diffs if abs(d) > 50)
    pnn50 = (nn50 / len(diffs)) * 100

    # Mean HR
    mean_hr = 60000.0 / mean_nn if mean_nn > 0 else None

    # Poincare SD1/SD2
    sd1 = math.sqrt(sum((d / math.sqrt(2)) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
    successive_sums = [rr_intervals[i] + rr_intervals[i + 1] for i in range(n - 1)]
    mean_ss = sum(successive_sums) / len(successive_sums)
    sd2 = math.sqrt(
        sum(((s - mean_ss) / math.sqrt(2)) ** 2 for s in successive_sums)
        / max(len(successive_sums) - 1, 1)
    )

    return {
        "sdnn": round(sdnn, 2),
        "rmssd": round(rmssd, 2),
        "pnn50": round(pnn50, 2),
        "mean_nn": round(mean_nn, 2),
        "mean_hr": round(mean_hr, 1) if mean_hr else None,
        "sd1": round(sd1, 2),
        "sd2": round(sd2, 2),
        "sd1_sd2_ratio": round(sd1 / sd2, 3) if sd2 > 0 else None,
        "beat_count": n,
    }


def _extract_rr_from_payload(payload: Dict[str, Any]) -> List[float]:
    """Extract RR intervals (ms) from a heartbeat_series payload."""
    rr_list = payload.get("rr_intervals", [])
    return [
        entry["rr_interval_ms"]
        for entry in rr_list
        if isinstance(entry, dict) and "rr_interval_ms" in entry
    ]


def _extract_rr_from_bpm(payload: Dict[str, Any]) -> List[float]:
    """Convert beat_to_beat_bpm array to RR intervals (ms)."""
    bpm_list = payload.get("beat_to_beat_bpm", [])
    rr = []
    for entry in bpm_list:
        bpm = entry.get("bpm") if isinstance(entry, dict) else entry
        if bpm and 20 < bpm < 250:
            rr.append(60000.0 / bpm)
    return rr


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/register")
def register(body: RegisterIn):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (user_id, last_seen_at)
                VALUES (%s, now())
                ON CONFLICT (user_id) DO UPDATE SET last_seen_at = now()
                """,
                (body.user_id,),
            )
    return {"registered": True, "user_id": body.user_id}


@app.post("/v1/ingest")
def ingest(body: IngestIn):
    # --- Validation ---
    if not body.samples:
        raise HTTPException(status_code=400, detail="samples is empty")

    if len(body.samples) > MAX_SAMPLES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many samples. Max {MAX_SAMPLES_PER_REQUEST} per request. "
            f"Got {len(body.samples)}. Please batch into smaller requests.",
        )

    for s in body.samples:
        if s.sample_type not in ALLOWED_SAMPLE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown sample_type: '{s.sample_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_SAMPLE_TYPES))}",
            )

    # Separate heartbeat_series samples for special handling
    regular_samples = []
    heartbeat_samples = []
    for s in body.samples:
        if s.sample_type == "heartbeat_series":
            heartbeat_samples.append(s)
        else:
            regular_samples.append(s)

    ingested = 0

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Upsert user
            cur.execute(
                """
                INSERT INTO users (user_id, last_seen_at)
                VALUES (%s, now())
                ON CONFLICT (user_id) DO UPDATE SET last_seen_at = now()
                """,
                (body.user_id,),
            )

            # --- Insert regular samples (no restrictions) ---
            if regular_samples:
                rows = [
                    (
                        body.user_id,
                        s.sample_type,
                        s.start_time,
                        s.end_time,
                        s.value,
                        s.unit,
                        s.source,
                        psycopg.types.json.Jsonb(s.payload)
                        if s.payload is not None
                        else None,
                    )
                    for s in regular_samples
                ]
                cur.executemany(
                    """
                    INSERT INTO health_samples
                      (user_id, sample_type, start_time, end_time, value, unit, source, payload)
                    VALUES
                      (%s, %s, %s::timestamptz, %s::timestamptz, %s, %s, %s, %s)
                    """,
                    rows,
                )
                ingested += len(regular_samples)

            # --- Heartbeat series: compute metrics, store metrics, keep max 2 raw ---
            for s in heartbeat_samples:
                computed_payload = s.payload  # default: store as-is

                if s.payload:
                    rr = _extract_rr_from_payload(s.payload)
                    if rr:
                        metrics = _compute_hrv_from_rr(rr)
                        if metrics:
                            # Store computed metrics instead of raw RR intervals
                            computed_payload = {
                                "computed_metrics": metrics,
                                "beat_count": s.payload.get("beat_count", len(rr) + 1),
                                "rr_intervals": s.payload.get("rr_intervals"),  # keep for now
                            }

                cur.execute(
                    """
                    INSERT INTO health_samples
                      (user_id, sample_type, start_time, end_time, value, unit, source, payload)
                    VALUES
                      (%s, %s, %s::timestamptz, %s::timestamptz, %s, %s, %s, %s)
                    """,
                    (
                        body.user_id,
                        s.sample_type,
                        s.start_time,
                        s.end_time,
                        s.value,
                        s.unit,
                        s.source,
                        psycopg.types.json.Jsonb(computed_payload)
                        if computed_payload is not None
                        else None,
                    ),
                )
                ingested += 1

            # --- Trim old heartbeat_series: keep only last N, strip RR from older ---
            if heartbeat_samples:
                # Get IDs of sessions to keep raw (most recent N)
                cur.execute(
                    """
                    SELECT id FROM health_samples
                    WHERE user_id = %s AND sample_type = 'heartbeat_series'
                    ORDER BY start_time DESC
                    LIMIT %s
                    """,
                    (body.user_id, MAX_HEARTBEAT_SESSIONS),
                )
                keep_ids = [r[0] for r in cur.fetchall()]

                if keep_ids:
                    # For older sessions: strip rr_intervals from payload, keep only metrics
                    cur.execute(
                        """
                        UPDATE health_samples
                        SET payload = payload - 'rr_intervals'
                        WHERE user_id = %s
                          AND sample_type = 'heartbeat_series'
                          AND payload IS NOT NULL
                          AND payload ? 'rr_intervals'
                          AND id NOT IN %s
                        """,
                        (body.user_id, tuple(keep_ids)),
                    )
                    trimmed = cur.rowcount
                    if trimmed > 0:
                        logger.info(
                            "Trimmed RR payload from %d old heartbeat_series for user %s",
                            trimmed,
                            body.user_id,
                        )

            # --- Also handle hrv_sdnn: compute metrics from BPM, store metrics ---
            # (hrv_sdnn samples are already in regular_samples; this is a post-process)
            # For hrv_sdnn with payload, compute metrics and add to payload
            cur.execute(
                """
                SELECT id, payload FROM health_samples
                WHERE user_id = %s
                  AND sample_type = 'hrv_sdnn'
                  AND payload IS NOT NULL
                  AND NOT (payload ? 'computed_metrics')
                ORDER BY start_time DESC
                LIMIT 100
                """,
                (body.user_id,),
            )
            sdnn_rows = cur.fetchall()
            for row_id, payload in sdnn_rows:
                if isinstance(payload, str):
                    payload = json.loads(payload)
                rr = _extract_rr_from_bpm(payload)
                if len(rr) >= 10:
                    metrics = _compute_hrv_from_rr(rr)
                    if metrics:
                        payload["computed_metrics"] = metrics
                        cur.execute(
                            """
                            UPDATE health_samples
                            SET payload = %s
                            WHERE id = %s
                            """,
                            (psycopg.types.json.Jsonb(payload), row_id),
                        )

    return {"ingested": ingested}


@app.get("/v1/summary")
def summary(user_id: str):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  avg(value) FILTER (WHERE sample_type='heart_rate') AS avg_hr,
                  avg(value) FILTER (WHERE sample_type='hrv') AS avg_hrv,
                  sum(value) FILTER (WHERE sample_type='steps') AS total_steps,
                  count(*) AS samples
                FROM health_samples
                WHERE user_id=%s
                """,
                (user_id,),
            )
            row = cur.fetchone()

    return {
        "user_id": user_id,
        "avg_heart_rate": row[0],
        "avg_hrv": row[1],
        "total_steps": row[2],
        "samples": row[3],
    }


@app.get("/v1/latest")
def latest(user_id: str):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sample_type, MAX(start_time) AS latest
                FROM health_samples
                WHERE user_id = %s
                GROUP BY sample_type
                """,
                (user_id,),
            )
            rows = cur.fetchall()

    if not rows:
        return {"user_id": user_id, "latest": None}

    latest_ts = max(r[1] for r in rows)

    return {
        "user_id": user_id,
        "latest": latest_ts.isoformat(),
        "by_type": {r[0]: r[1].isoformat() for r in rows},
    }


@app.post("/v1/cleanup")
def cleanup(user_id: str = Query(...), days: int = Query(default=365)):
    """Delete samples older than N days for a user."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM health_samples
                WHERE user_id = %s
                  AND start_time < NOW() - make_interval(days => %s)
                """,
                (user_id, days),
            )
            deleted = cur.rowcount
    return {"deleted": deleted, "user_id": user_id, "retention_days": days}
