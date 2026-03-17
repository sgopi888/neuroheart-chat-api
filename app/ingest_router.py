from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional

import psycopg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["ingest"])

# --- Limits ---
MAX_SAMPLES_PER_REQUEST = 5000
MAX_HEARTBEAT_SESSIONS = 5  # keep last 5 sessions with raw RR intervals for the AI

ALLOWED_SAMPLE_TYPES = {
    "heart_rate",
    "hrv",
    "steps",
    "sleep",
    "hrv_sdnn",
    "heartbeat_series",
}

# --- Schemas ---

class SampleIn(BaseModel):
    sample_type: str
    start_time: str
    end_time: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    user_id: str = Field(min_length=8, max_length=128)
    samples: List[SampleIn]

# --- HRV Logic ---

def _compute_hrv_from_rr(rr_intervals: List[float]) -> Optional[Dict[str, Any]]:
    if len(rr_intervals) < 10: return None
    n = len(rr_intervals)
    mean_nn = sum(rr_intervals) / n
    variance = sum((x - mean_nn) ** 2 for x in rr_intervals) / (n - 1)
    sdnn = math.sqrt(variance)
    diffs = [rr_intervals[i+1] - rr_intervals[i] for i in range(n-1)]
    rmssd = math.sqrt(sum(d**2 for d in diffs)/len(diffs))
    return {
        "sdnn": round(sdnn, 2),
        "rmssd": round(rmssd, 2),
        "beat_count": n,
        "mean_hr": round(60000.0 / mean_nn, 1) if mean_nn > 0 else None
    }

def _extract_rr_from_payload(payload: Dict[str, Any]) -> List[float]:
    rr_list = payload.get("rr_intervals", [])
    return [entry["rr_interval_ms"] for entry in rr_list if isinstance(entry, dict) and "rr_interval_ms" in entry]

def _extract_rr_from_bpm(payload: Dict[str, Any]) -> List[float]:
    bpm_list = payload.get("beat_to_beat_bpm", [])
    return [60000.0 / b.get("bpm") for b in bpm_list if isinstance(b, dict) and b.get("bpm", 0) > 20]

# --- Endpoints ---

@router.post("/ingest")
async def ingest_health_data(body: IngestRequest):
    if not body.samples:
        raise HTTPException(status_code=400, detail="Samples list is empty")
    
    # 1. Separate heartbeat series
    regular_samples = [s for s in body.samples if s.sample_type != "heartbeat_series"]
    heartbeat_samples = [s for s in body.samples if s.sample_type == "heartbeat_series"]

    ingested = 0
    db_url = settings.database_url
    
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # 2. Ingest regular samples
            if regular_samples:
                rows = [
                    (body.user_id, s.sample_type, s.start_time, s.end_time, s.value, s.unit, s.source, 
                     json.dumps(s.payload) if s.payload else None)
                    for s in regular_samples
                ]
                cur.executemany(
                    "INSERT INTO health_samples (user_id, sample_type, start_time, end_time, value, unit, source, payload) "
                    "VALUES (%s, %s, %s::timestamptz, %s::timestamptz, %s, %s, %s, %s::jsonb)",
                    rows
                )
                ingested += len(regular_samples)

            # 3. Handle Heartbeat Series with auto-metrics
            for s in heartbeat_samples:
                computed_payload = s.payload
                if s.payload:
                    rr = _extract_rr_from_payload(s.payload)
                    metrics = _compute_hrv_from_rr(rr)
                    if metrics:
                        computed_payload["computed_metrics"] = metrics
                
                cur.execute(
                    "INSERT INTO health_samples (user_id, sample_type, start_time, end_time, value, unit, source, payload) "
                    "VALUES (%s, %s, %s::timestamptz, %s::timestamptz, %s, %s, %s, %s::jsonb)",
                    (body.user_id, s.sample_type, s.start_time, s.end_time, s.value, s.unit, s.source, json.dumps(computed_payload))
                )
                ingested += 1

            # 4. Handle HRV SDNN Metadata (Query 1)
            # Post-process the newly added regular samples of type hrv_sdnn to add metrics
            cur.execute(
                "SELECT id, payload FROM health_samples WHERE user_id = %s AND sample_type = 'hrv_sdnn' AND payload IS NOT NULL AND NOT (payload ? 'computed_metrics')",
                (body.user_id,)
            )
            for row_id, payload in cur.fetchall():
                rr = _extract_rr_from_bpm(payload)
                metrics = _compute_hrv_from_rr(rr)
                if metrics:
                    payload["computed_metrics"] = metrics
                    cur.execute("UPDATE health_samples SET payload = %s::jsonb WHERE id = %s", (json.dumps(payload), row_id))

    return {"status": "success", "ingested": ingested}
