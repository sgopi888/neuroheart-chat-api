from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.db import get_engine


def assert_conversation_owner(user_uid: str, conversation_id: str) -> None:
    """Raise LookupError if conversation doesn't belong to user."""
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text("SELECT 1 FROM conversations WHERE id = :cid AND user_uid = :uid"),
            {"cid": conversation_id, "uid": user_uid},
        ).fetchone()
    if row is None:
        raise LookupError("conversation_not_found")


def create_conversation(user_uid: str, title: Optional[str]) -> Dict[str, Any]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO conversations (user_uid, title)
                VALUES (:uid, :title)
                RETURNING id::text AS conversation_id, created_at::text AS created_at
                """
            ),
            {"uid": user_uid, "title": title},
        ).fetchone()
    return {"conversation_id": row.conversation_id, "created_at": row.created_at}


def list_conversations(user_uid: str, limit: int = 50) -> List[Dict[str, Any]]:
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id::text AS conversation_id,
                       title,
                       updated_at::text AS updated_at
                FROM conversations
                WHERE user_uid = :uid AND is_archived = FALSE
                ORDER BY updated_at DESC
                LIMIT :lim
                """
            ),
            {"uid": user_uid, "lim": limit},
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def fetch_history(
    user_uid: str,
    conversation_id: str,
    limit: int = 50,
    before_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    assert_conversation_owner(user_uid, conversation_id)
    eng = get_engine()
    with eng.begin() as conn:
        if before_id is not None:
            rows = conn.execute(
                text(
                    """
                    SELECT id, role, content, created_at::text AS created_at
                    FROM chat_messages
                    WHERE conversation_id = :cid AND user_uid = :uid AND id < :bid
                    ORDER BY created_at DESC
                    LIMIT :lim
                    """
                ),
                {"cid": conversation_id, "uid": user_uid, "bid": before_id, "lim": limit},
            ).fetchall()
        else:
            rows = conn.execute(
                text(
                    """
                    SELECT id, role, content, created_at::text AS created_at
                    FROM chat_messages
                    WHERE conversation_id = :cid AND user_uid = :uid
                    ORDER BY created_at DESC
                    LIMIT :lim
                    """
                ),
                {"cid": conversation_id, "uid": user_uid, "lim": limit},
            ).fetchall()
    items = [dict(r._mapping) for r in rows]
    return list(reversed(items))  # return ascending for display / prompting


def insert_message(
    user_uid: str,
    conversation_id: str,
    role: str,
    content: str,
    model: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    assert_conversation_owner(user_uid, conversation_id)
    meta_json = json.dumps(metadata) if metadata else None
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO chat_messages (conversation_id, user_uid, role, content, model, metadata)
                VALUES (:cid, :uid, :role, :content, :model, CAST(:meta AS jsonb))
                """
            ),
            {
                "cid": conversation_id,
                "uid": user_uid,
                "role": role,
                "content": content,
                "model": model,
                "meta": meta_json,
            },
        )
        conn.execute(
            text(
                "UPDATE conversations SET updated_at = now() WHERE id = :cid AND user_uid = :uid"
            ),
            {"cid": conversation_id, "uid": user_uid},
        )


def count_messages(conversation_id: str) -> int:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text("SELECT COUNT(*) AS cnt FROM chat_messages WHERE conversation_id = :cid"),
            {"cid": conversation_id},
        ).fetchone()
    return int(row.cnt)


def get_or_create_summary(conversation_id: str, user_uid: str) -> Dict[str, Any]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text(
                """
                SELECT summary, summarized_through_message_id
                FROM conversation_summaries
                WHERE conversation_id = :cid
                """
            ),
            {"cid": conversation_id},
        ).fetchone()
        if row is None:
            conn.execute(
                text(
                    """
                    INSERT INTO conversation_summaries (conversation_id, user_uid)
                    VALUES (:cid, :uid)
                    ON CONFLICT (conversation_id) DO NOTHING
                    """
                ),
                {"cid": conversation_id, "uid": user_uid},
            )
            return {"summary": "", "summarized_through_message_id": None}
    return dict(row._mapping)


def update_summary(
    conversation_id: str,
    summarized_through_id: int,
    summary_text: str,
) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE conversation_summaries
                SET summary = :s,
                    summarized_through_message_id = :mid,
                    updated_at = now()
                WHERE conversation_id = :cid
                """
            ),
            {"s": summary_text, "mid": summarized_through_id, "cid": conversation_id},
        )


def fetch_messages_for_summarization(
    conversation_id: str,
    after_id: Optional[int],
    before_id: int,
) -> List[Dict[str, Any]]:
    """Fetch messages older than before_id and newer than after_id for summarization."""
    eng = get_engine()
    with eng.begin() as conn:
        if after_id is not None:
            rows = conn.execute(
                text(
                    """
                    SELECT id, role, content
                    FROM chat_messages
                    WHERE conversation_id = :cid AND id > :aid AND id < :bid
                    ORDER BY id ASC
                    """
                ),
                {"cid": conversation_id, "aid": after_id, "bid": before_id},
            ).fetchall()
        else:
            rows = conn.execute(
                text(
                    """
                    SELECT id, role, content
                    FROM chat_messages
                    WHERE conversation_id = :cid AND id < :bid
                    ORDER BY id ASC
                    """
                ),
                {"cid": conversation_id, "bid": before_id},
            ).fetchall()
    return [dict(r._mapping) for r in rows]


def get_cross_chat_profile(user_uid: str) -> str:
    """Get the cross-chat user profile. Returns empty string if none exists."""
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text("SELECT profile FROM user_cross_chat_profiles WHERE user_uid = :uid"),
            {"uid": user_uid},
        ).fetchone()
    return row.profile if row else ""


def upsert_cross_chat_profile(user_uid: str, profile: str) -> None:
    """Create or update the cross-chat user profile."""
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO user_cross_chat_profiles (user_uid, profile)
                VALUES (:uid, :profile)
                ON CONFLICT (user_uid) DO UPDATE SET profile = :profile, updated_at = now()
                """
            ),
            {"uid": user_uid, "profile": profile},
        )


def fetch_recent_message_ids(conversation_id: str, n: int = 20) -> List[int]:
    """Return the IDs of the most recent N messages (ascending)."""
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id FROM chat_messages
                WHERE conversation_id = :cid
                ORDER BY id DESC
                LIMIT :n
                """
            ),
            {"cid": conversation_id, "n": n},
        ).fetchall()
    ids = [r.id for r in rows]
    return list(reversed(ids))


# ── Audio Narrations ──────────────────────────────────────────────


def insert_audio_narration(
    user_uid: str,
    conversation_id: str,
    session_id: str,
    meditation_type: str,
    audio_type: str,
    file_path: str,
    duration_seconds: int | None = None,
    title: str | None = None,
    metadata: dict | None = None,
) -> Dict:
    """Insert an audio narration record and return {id, created_at}."""
    import json

    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text(
                """
                INSERT INTO audio_narrations
                    (user_uid, conversation_id, session_id, meditation_type,
                     audio_type, file_path, duration_seconds, title, metadata)
                VALUES
                    (:uid, CAST(:cid AS UUID), CAST(:sid AS UUID), :mtype,
                     :atype, :fpath, :dur, :title, CAST(:meta AS JSONB))
                RETURNING id, created_at
                """
            ),
            {
                "uid": user_uid,
                "cid": conversation_id,
                "sid": session_id,
                "mtype": meditation_type,
                "atype": audio_type,
                "fpath": file_path,
                "dur": duration_seconds,
                "title": title,
                "meta": json.dumps(metadata) if metadata else None,
            },
        ).fetchone()
    return {"id": str(row.id), "created_at": str(row.created_at)}


def list_audio_narrations(user_uid: str, limit: int = 25) -> List[Dict]:
    """List audio narrations for a user, ordered by newest first."""
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT id, session_id, conversation_id, meditation_type,
                       audio_type, file_path, duration_seconds, title,
                       metadata, created_at
                FROM audio_narrations
                WHERE user_uid = :uid
                ORDER BY created_at DESC
                LIMIT :lim
                """
            ),
            {"uid": user_uid, "lim": limit},
        ).fetchall()
    return [dict(r._mapping) for r in rows]


def delete_audio_narration(narration_id: str, user_uid: str) -> str | None:
    """Delete an audio narration by id (with ownership check). Returns file_path or None."""
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(
            text(
                """
                DELETE FROM audio_narrations
                WHERE id = CAST(:nid AS UUID) AND user_uid = :uid
                RETURNING file_path
                """
            ),
            {"nid": narration_id, "uid": user_uid},
        ).fetchone()
    return row.file_path if row else None


def enforce_audio_limit(user_uid: str, max_count: int = 25) -> List[str]:
    """Delete oldest narrations if user has more than max_count. Returns deleted file_paths."""
    eng = get_engine()
    with eng.begin() as conn:
        count_row = conn.execute(
            text("SELECT COUNT(*) AS cnt FROM audio_narrations WHERE user_uid = :uid"),
            {"uid": user_uid},
        ).fetchone()
        excess = count_row.cnt - max_count
        if excess <= 0:
            return []
        rows = conn.execute(
            text(
                """
                DELETE FROM audio_narrations
                WHERE id IN (
                    SELECT id FROM audio_narrations
                    WHERE user_uid = :uid
                    ORDER BY created_at ASC
                    LIMIT :excess
                )
                RETURNING file_path
                """
            ),
            {"uid": user_uid, "excess": excess},
        ).fetchall()
    return [r.file_path for r in rows]
