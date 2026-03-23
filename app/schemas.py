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
    calendar_change: bool = False
    calendar_command: Optional[str] = None
    conversation_id: str
    reply: str
    used_context: bool
    hrv_range: str
    rag_k: int
    generate_meditation: bool = False
    meditation_audio_url: Optional[str] = None
    meditation_title: Optional[str] = None
    meditation_session_id: Optional[str] = None


class PracticeRequest(BaseModel):
    user_uid: str
    conversation_id: str
    mood: str
    depth: Optional[str] = None
    duration: int
    session_type: str = "breathing"


class PracticeResponse(BaseModel):
    conversation_id: str
    script: str
    title: str


# ── Meditation Audio ──────────────────────────────────────────────

class MusicConfig(BaseModel):
    mood: Optional[str] = None
    style: Optional[str] = None
    additional_context: Optional[str] = None


class GenerateMeditationRequest(BaseModel):
    user_uid: str
    conversation_id: str
    mood: str
    depth: Optional[str] = None
    duration: int  # minutes
    session_type: str = "meditation"
    music_config: Optional[MusicConfig] = None


class GenerateMeditationResponse(BaseModel):
    session_id: str
    conversation_id: str
    script: str
    title: str
    audio_url: str  # merged ready-to-play URL
    meditation_type: str  # 'short' | 'medium' | 'deep'


class AudioUploadRequest(BaseModel):
    user_uid: str
    session_id: str
    conversation_id: str
    meditation_type: str
    audio_base64: str
    duration_seconds: Optional[int] = None
    title: Optional[str] = None
    metadata: Optional[dict] = None


class AudioUploadResponse(BaseModel):
    id: str
    session_id: str
    created_at: str


class AudioNarrationItem(BaseModel):
    id: str
    session_id: str
    conversation_id: str
    meditation_type: str
    audio_type: str
    audio_url: str
    duration_seconds: Optional[int] = None
    title: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: str


class AudioListResponse(BaseModel):
    narrations: List[AudioNarrationItem]
