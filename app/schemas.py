from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class CreateConversationRequest(BaseModel):
    user_uid: str
    title: Optional[str] = None


class CreateConversationResponse(BaseModel):
    conversation_id: str
    created_at: str


class ConversationItem(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    updated_at: str


class ListConversationsResponse(BaseModel):
    conversations: List[ConversationItem]


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class HistoryResponse(BaseModel):
    conversation_id: str
    messages: List[MessageItem]


class ChatRequest(BaseModel):
    user_uid: str
    conversation_id: str
    message: str = Field(min_length=1, max_length=4000)
    hrv_range: str = "7d"  # 1d | 7d | 30d | 6m


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    used_context: bool
    hrv_range: str
    rag_k: int
