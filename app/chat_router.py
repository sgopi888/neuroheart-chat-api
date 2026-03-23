from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from app.chat_service import chat_once
from app.config import settings
from app.history_repository import (
    create_conversation,
    fetch_history,
    insert_message,
    list_conversations,
)
from app.meditation_service import generate_meditation
from app.rate_limit import allow
from app.schemas import (
    ChatRequest,
    ChatResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    HistoryResponse,
    ListConversationsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["chat"])


def _require_app_token(x_app_token: Optional[str]) -> None:
    """Reject requests with a missing or invalid app token."""
    if settings.app_token and x_app_token != settings.app_token:
        raise HTTPException(status_code=403, detail="forbidden")


@router.post("/conversations", response_model=CreateConversationResponse)
def create_conv(
    req: CreateConversationRequest,
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    _require_app_token(x_app_token)
    try:
        return create_conversation(req.user_uid, req.title)
    except Exception:
        raise HTTPException(status_code=500, detail="create_failed")


@router.get("/conversations", response_model=ListConversationsResponse)
def list_conv(
    user_uid: str,
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    _require_app_token(x_app_token)
    items = list_conversations(user_uid)
    return {"conversations": items}


@router.get("/history", response_model=HistoryResponse)
def history(
    user_uid: str,
    conversation_id: str,
    limit: int = 50,
    before_id: Optional[int] = None,
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    _require_app_token(x_app_token)
    try:
        msgs = fetch_history(user_uid, conversation_id, limit=limit, before_id=before_id)
        return {"conversation_id": conversation_id, "messages": msgs}
    except LookupError:
        raise HTTPException(status_code=404, detail="not_found")


_MEDITATION_TAG = "[GENERATE_MEDITATION]"

# Keywords in user message that indicate a meditation request
_MEDITATION_USER_KEYWORDS = [
    "generate meditation", "generate a meditation", "create a meditation",
    "make me a meditation", "make a meditation", "breathing exercise",
    "generate breathing", "create a breathing", "mindfulness exercise",
    "guided meditation", "can you make me a meditation",
    "create meditation", "make meditation",
]


def _detect_meditation_request(user_message: str, llm_reply: str) -> bool:
    """Check if the user asked for or the LLM suggested generating a meditation."""
    # Check LLM reply for the tag
    if _MEDITATION_TAG in llm_reply:
        return True
    # Check user message for keywords
    msg_lower = user_message.lower()
    return any(kw in msg_lower for kw in _MEDITATION_USER_KEYWORDS)


@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    _require_app_token(x_app_token)

    if not allow(req.user_uid):
        raise HTTPException(status_code=429, detail="rate_limited")

    try:
        out = await chat_once(
            req.user_uid,
            req.conversation_id,
            req.message,
            req.hrv_range,
            rag_k=3,
        )

        reply = out["reply"]

        # Detect calendar action in LLM reply
        cal_keywords = [
            "add to your calendar", "i'll add", "i've added", "i'll schedule",
            "i've scheduled", "i'll cancel", "i'll remove", "i'll move",
            "i've moved", "i'll update", "i've updated", "to your calendar",
            "i'll create", "i've created", "i'll delete", "i've deleted",
            "added to your calendar", "scheduled for", "event has been"
        ]
        reply_lower = reply.lower()
        is_calendar = any(kw in reply_lower for kw in cal_keywords)

        # Detect meditation generation request
        is_meditation = _detect_meditation_request(req.message, reply)

        # Strip the tag from the reply shown to user
        if _MEDITATION_TAG in reply:
            reply = reply.replace(_MEDITATION_TAG, "").strip()

        meditation_audio_url = None
        meditation_title = None
        meditation_session_id = None

        # Auto-generate meditation if triggered
        if is_meditation:
            try:
                med_result = await generate_meditation(
                    user_uid=req.user_uid,
                    conversation_id=req.conversation_id,
                    mood="calm",
                    depth=None,
                    duration=5,
                    session_type="meditation",
                )
                meditation_audio_url = med_result["audio_url"]
                meditation_title = med_result["title"]
                meditation_session_id = med_result["session_id"]

                # Insert meditation link as assistant message in chat history
                med_message = (
                    f"Here's your meditation: {meditation_title}\n"
                    f"[MEDITATION_AUDIO:{meditation_audio_url}]"
                )
                insert_message(
                    user_uid=req.user_uid,
                    conversation_id=req.conversation_id,
                    role="assistant",
                    content=med_message,
                    metadata={
                        "type": "meditation_link",
                        "session_id": meditation_session_id,
                        "audio_url": meditation_audio_url,
                        "title": meditation_title,
                    },
                )
            except Exception as exc:
                logger.exception("Auto meditation generation failed: %s", exc)
                # Don't fail the whole chat response if meditation fails

        return {
            "conversation_id": req.conversation_id,
            "reply": reply,
            "calendar_change": is_calendar,
            "calendar_command": req.message if is_calendar else None,
            "used_context": out["used_context"],
            "hrv_range": req.hrv_range,
            "rag_k": out["rag_k"],
            "generate_meditation": is_meditation,
            "meditation_audio_url": meditation_audio_url,
            "meditation_title": meditation_title,
            "meditation_session_id": meditation_session_id,
        }
    except LookupError:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    except Exception as exc:
        logger.exception("chat_once failed: %s", exc)
        raise HTTPException(status_code=500, detail="chat_failed")
