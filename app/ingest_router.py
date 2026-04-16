from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import psycopg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.hrv_bpm_per_min import process_bpm_session, save_session_results, load_cross_session_baseline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["ingest"])

from app.config import settings as _cfg

MAX_SAMPLES_PER_REQUEST = _cfg.ingest_max_samples
MAX_HEARTBEAT_SESSIONS = _cfg.ingest_max_heartbeat_sessions

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
    model_config = {"extra": "allow"}

    sample_type: str
    start_time: str
    end_time: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _map_sdnn_value(self):
        """iOS sends 'sdnn_value' instead of 'value' for hrv_sdnn samples."""
        if self.value is None and self.sample_type == "hrv_sdnn":
            sdnn = (self.model_extra or {}).get("sdnn_value")
            if sdnn is not None:
                self.value = float(sdnn)
        return self

class IngestRequest(BaseModel):
    user_id: str = Field(min_length=8, max_length=128)
    samples: List[SampleIn]

from app.hrv_utils import compute_hrv_from_rr as _compute_hrv_from_rr_shared

def _compute_hrv_from_rr(rr_intervals: List[float]) -> Optional[Dict[str, Any]]:
    return _compute_hrv_from_rr_shared(rr_intervals, min_count=10)

def _extract_rr_from_payload(payload: Dict[str, Any]) -> List[float]:
    rr_list = payload.get("rr_intervals", [])
    return [entry["rr_interval_ms"] for entry in rr_list if isinstance(entry, dict) and "rr_interval_ms" in entry]

def _extract_rr_from_bpm(payload: Dict[str, Any]) -> List[float]:
    bpm_list = payload.get("beat_to_beat_bpm") or payload.get("beat_to_beat_metadata") or []
    return [60000.0 / b.get("bpm") for b in bpm_list if isinstance(b, dict) and b.get("bpm", 0) > 20]

# --- Endpoints ---

@router.post("/ingest")
async def ingest_health_data(body: IngestRequest):
    if not body.samples:
        raise HTTPException(status_code=400, detail="Samples list is empty")

    # Log incoming sample breakdown for debugging
    type_counts: Dict[str, int] = {}
    null_values = []
    null_payloads = []
    for s in body.samples:
        type_counts[s.sample_type] = type_counts.get(s.sample_type, 0) + 1
        if s.value is None and s.sample_type in ("hrv_sdnn", "hrv", "heart_rate"):
            null_values.append(s.sample_type)
        if s.payload is None and s.sample_type == "heartbeat_series":
            null_payloads.append(s.sample_type)
    logger.info("Ingest user=%s counts=%s null_values=%d null_payloads=%d",
                body.user_id[:12], type_counts, len(null_values), len(null_payloads))

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

            # 4. Run BPM-to-calm-score pipeline on heartbeat series
            for s in heartbeat_samples:
                if not s.payload:
                    continue
                # Watch sends rr_intervals; hrv_sdnn sends beat_to_beat_bpm
                bpm_list = (s.payload.get("rr_intervals")
                            or s.payload.get("beat_to_beat_bpm")
                            or [])
                if len(bpm_list) < 60:
                    logger.info("Skipping BPM pipeline: only %d samples", len(bpm_list))
                    continue
                try:
                    cross_bl = load_cross_session_baseline(body.user_id)
                    snapshots, summary = process_bpm_session(bpm_list, cross_bl)
                    if snapshots:
                        saved = save_session_results(body.user_id, s.start_time, snapshots, summary)
                        logger.info("BPM pipeline: %d snapshots, avg_calm=%.1f, saved=%d rows",
                                    len(snapshots), summary.avg_calm_score, saved)
                except Exception:
                    logger.exception("BPM pipeline failed for user=%s", body.user_id[:12])

            # 5. Handle HRV SDNN Metadata (Query 1)
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
