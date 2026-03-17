from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.db import get_engine

from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/calendar", tags=["calendar-sync"])


# Schemas

class CalendarSyncEvent(BaseModel):
    title: str
    start_time: str
    end_time: str
    is_all_day: bool = False
    location: Optional[str] = None
    notes: Optional[str] = None
    calendar_name: Optional[str] = None
    is_recurring: bool = False


class CalendarSyncRequest(BaseModel):
    user_uid: str
    events: List[CalendarSyncEvent]
    sync_days: int = 7
    timezone: str = "UTC"


class CalendarSyncResponse(BaseModel):
    status: str
    events_stored: int


# Endpoint

def _require_app_token(x_app_token: Optional[str]) -> None:
    if settings.app_token and x_app_token != settings.app_token:
        raise HTTPException(status_code=403, detail="forbidden")


@router.post("/sync", response_model=CalendarSyncResponse)
def sync_calendar(
    req: CalendarSyncRequest,
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    _require_app_token(x_app_token)

    events_json = json.dumps([e.model_dump() for e in req.events], default=str)

    upsert_calendar_context(
        user_uid=req.user_uid,
        events_json=events_json,
        sync_days=req.sync_days,
        timezone=req.timezone,
    )

    logger.info(
        "Calendar sync: user=%s events=%d days=%d",
        req.user_uid[:12], len(req.events), req.sync_days,
    )
    return {"status": "ok", "events_stored": len(req.events)}


# DB helpers

def upsert_calendar_context(
    user_uid: str,
    events_json: str,
    sync_days: int,
    timezone: str,
) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO user_calendar_context
                (user_uid, events_json, sync_days, timezone, synced_at, updated_at)
            VALUES (:uid, CAST(:events AS jsonb), :days, :tz, now(), now())
            ON CONFLICT (user_uid) DO UPDATE SET
                events_json = CAST(:events AS jsonb),
                sync_days   = :days,
                timezone    = :tz,
                synced_at   = now(),
                updated_at  = now()
        """), {
            "uid": user_uid,
            "events": events_json,
            "days": sync_days,
            "tz": timezone,
        })


def get_calendar_context(user_uid: str) -> Optional[Dict[str, Any]]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("""
            SELECT events_json, sync_days, timezone, synced_at::text AS synced_at
            FROM user_calendar_context
            WHERE user_uid = :uid
        """), {"uid": user_uid}).fetchone()

    if row is None:
        return None

    return {
        "events": row.events_json if isinstance(row.events_json, list)
                  else json.loads(row.events_json),
        "sync_days": row.sync_days,
        "timezone": row.timezone,
        "synced_at": row.synced_at,
    }


# Context Formatter

def format_calendar_context(user_uid: str) -> str:
    ctx = get_calendar_context(user_uid)
    if not ctx or not ctx["events"]:
        return ""

    lines: List[str] = []
    for ev in ctx["events"]:
        start = ev.get("start_time", "")[:16].replace("T", " ")
        end = ev.get("end_time", "")[:16].replace("T", " ")

        if ev.get("is_all_day"):
            date_part = f"{start[:10]} (all day)"
        else:
            if start[:10] == end[:10]:
                date_part = f"{start} - {end[11:]}"
            else:
                date_part = f"{start} - {end}"

        parts = [date_part, ev.get("title", "Untitled")]
        if ev.get("is_recurring"):
            parts.append("(recurring)")
        if ev.get("location"):
            parts.append(f"@ {ev['location']}")

        lines.append(" | ".join(parts))

    header = f"CALENDAR_CONTEXT (past {ctx['sync_days']} days + next {ctx['sync_days']} days, tz={ctx['timezone']}):"
    return header + "\n" + "\n".join(lines)
