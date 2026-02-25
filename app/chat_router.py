from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException

from app.chat_service import chat_once
from app.config import settings
from app.history_repository import create_conversation, fetch_history, list_conversations
from app.rate_limit import allow
from app.schemas import (
    ChatRequest,
    ChatResponse,
    CreateConversationRequest,
    CreateConversationResponse,
    HistoryResponse,
    ListConversationsResponse,
)

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
        return {
            "conversation_id": req.conversation_id,
            "reply": out["reply"],
            "used_context": out["used_context"],
            "hrv_range": req.hrv_range,
            "rag_k": out["rag_k"],
        }
    except LookupError:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="chat_failed")
