"""Repository for async meditation generation jobs.

Backs the POST /v1/practice/generate-meditation-async + GET /v1/practice/job/{id}
flow, so the client can start a long generation and re-attach by job_id after the
app is backgrounded, killed, or the phone is locked/rebooted.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from sqlalchemy import text

from app.db import get_engine


def create_job(
    user_uid: str,
    conversation_id: str,
    mood: str,
    depth: Optional[str],
    duration: int,
    session_type: str,
    music_config: Optional[dict],
) -> str:
    """Insert a new pending job and return its job_id."""
    music_json = json.dumps(music_config) if music_config else None
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO meditation_jobs
                    (user_uid, conversation_id, mood, depth, duration,
                     session_type, music_config, status)
                VALUES
                    (:uid, :cid, :mood, :depth, :duration,
                     :stype, CAST(:music AS jsonb), 'pending')
                RETURNING id
                """
            ),
            {
                "uid": user_uid,
                "cid": conversation_id,
                "mood": mood,
                "depth": depth,
                "duration": duration,
                "stype": session_type,
                "music": music_json,
            },
        ).fetchone()
    return str(row._mapping["id"])


def mark_running(job_id: str) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE meditation_jobs
                SET status = 'running', updated_at = now()
                WHERE id = :jid
                """
            ),
            {"jid": job_id},
        )


def complete_job(
    job_id: str,
    session_id: Optional[str],
    title: Optional[str],
    audio_url: Optional[str],
    meditation_type: Optional[str],
) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE meditation_jobs
                SET status = 'ready',
                    session_id = CAST(:sid AS uuid),
                    title = :title,
                    audio_url = :audio_url,
                    meditation_type = :mtype,
                    updated_at = now()
                WHERE id = :jid
                """
            ),
            {
                "jid": job_id,
                "sid": session_id,
                "title": title,
                "audio_url": audio_url,
                "mtype": meditation_type,
            },
        )


def fail_job(job_id: str, error: str) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE meditation_jobs
                SET status = 'failed', error = :error, updated_at = now()
                WHERE id = :jid
                """
            ),
            {"jid": job_id, "error": error[:1000]},
        )


def get_job(job_id: str, user_uid: str) -> Optional[Dict[str, Any]]:
    """Fetch a job scoped to its owner. Returns None if not found / not owned."""
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT id, user_uid, conversation_id, status,
                       session_id, title, audio_url, meditation_type, error,
                       created_at, updated_at
                FROM meditation_jobs
                WHERE id = :jid AND user_uid = :uid
                """
            ),
            {"jid": job_id, "uid": user_uid},
        ).fetchone()
    if row is None:
        return None
    return dict(row._mapping)
