"""Account deletion (Apple remediation Phase 5, Task 5.4 / 5.5).

Permanently purges a user's data from the active Postgres database in a single
transaction, then writes a minimal, content-free security-retention record so
fraud / abuse / legal-compliance needs can be met without keeping wellness data.

Confirmed schema (verified on the VPS):

  Cascades from users(user_id) ON DELETE CASCADE:
    - health_samples        (user_id)
    - mindfulness_sessions   (user_id)

  No FK — must be deleted explicitly (keyed by user_uid):
    - chat_messages
    - conversations
    - conversation_summaries
    - audio_narrations
    - user_calendar_context
    - user_cross_chat_profiles

Deletion order: write the retention row, delete the explicit (user_uid) content
tables, then delete the users row (which cascades health_samples and
mindfulness_sessions). All in one transaction so a failure rolls back cleanly.

The user_id used by Apple Sign-In ("sub" claim) is the same value stored as both
users.user_id and the *_uid columns in the content tables, so a single identifier
drives the whole purge.

NOT handled here (documented TODOs — run / wire separately):
  * Apple Sign-In token revocation. This service authenticates via the Apple ID
    token directly and stores no Apple refresh token, so there is nothing to
    revoke server-side here. If a refresh token is ever stored, call Apple's
    POST https://appleid.apple.com/auth/revoke. (There is no Firebase Auth in
    this backend.)
  * Vector memories (Qdrant) — if long-term user memories are persisted to a
    Qdrant collection, purge the user's points/payload here. Verify collection
    naming on the VPS before enabling.
  * On-disk audio files under AUDIO_STORAGE_DIR referenced by audio_narrations —
    DB rows are removed; orphaned files should be removed by a separate sweep.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text

from app.db import get_engine

logger = logging.getLogger(__name__)

# Content tables keyed by user_uid that do NOT cascade from users — delete first.
_USER_UID_TABLES = (
    "chat_messages",
    "conversations",
    "conversation_summaries",
    "audio_narrations",
    "user_calendar_context",
    "user_cross_chat_profiles",
)


def delete_account(user_uid: str, reason: Optional[str] = None) -> dict:
    """Purge all data for ``user_uid`` and record a retention stub.

    Returns a summary dict with per-table delete counts. Idempotent: deleting an
    already-deleted account simply removes nothing and still records a retention
    row (rows are keyed by (user_id, deleted_at)).
    """
    if not user_uid:
        raise ValueError("user_uid is required")

    deleted_counts: dict[str, int] = {}
    eng = get_engine()

    with eng.begin() as conn:
        # 1. Minimal, content-free security-retention record (Task 5.5).
        conn.execute(
            text(
                """
                INSERT INTO deleted_accounts (user_id, reason)
                VALUES (:uid, :reason)
                """
            ),
            {"uid": user_uid, "reason": reason},
        )

        # 2. Explicit content tables (no cascade).
        for table in _USER_UID_TABLES:
            res = conn.execute(
                text(f"DELETE FROM {table} WHERE user_uid = :uid"),
                {"uid": user_uid},
            )
            deleted_counts[table] = res.rowcount or 0

        # 3. The users row — cascades health_samples + mindfulness_sessions.
        res = conn.execute(
            text("DELETE FROM users WHERE user_id = :uid"),
            {"uid": user_uid},
        )
        deleted_counts["users"] = res.rowcount or 0

    logger.info(
        "account deleted — user_uid=%s counts=%s",
        user_uid[:12],
        deleted_counts,
    )
    return {"status": "deleted", "user_uid": user_uid, "deleted": deleted_counts}
