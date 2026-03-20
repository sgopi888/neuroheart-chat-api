"""Meditation audio generation endpoints."""

from __future__ import annotations

import base64
import datetime
import logging
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import settings
from app.history_repository import (
    delete_audio_narration,
    insert_audio_narration,
    list_audio_narrations,
)
from app.meditation_service import generate_meditation
from app.schemas import (
    AudioListResponse,
    AudioNarrationItem,
    AudioUploadRequest,
    AudioUploadResponse,
    GenerateMeditationRequest,
    GenerateMeditationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/practice", tags=["meditation"])


def _require_app_token(x_app_token: Optional[str]) -> None:
    if settings.app_token and x_app_token != settings.app_token:
        raise HTTPException(status_code=403, detail="forbidden")


@router.post("/generate-meditation", response_model=GenerateMeditationResponse)
async def generate_meditation_endpoint(
    req: GenerateMeditationRequest,
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    """Generate SSML meditation script, voice narration, ambient music, and merge."""
    _require_app_token(x_app_token)

    try:
        music_config = None
        if req.music_config:
            music_config = req.music_config.model_dump(exclude_none=True)

        result = await generate_meditation(
            user_uid=req.user_uid,
            conversation_id=req.conversation_id,
            mood=req.mood,
            depth=req.depth,
            duration=req.duration,
            session_type=req.session_type,
            music_config=music_config,
        )
        return result

    except LookupError:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    except RuntimeError as exc:
        logger.error("Meditation generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("generate_meditation failed: %s", exc)
        raise HTTPException(status_code=500, detail="meditation_generation_failed")


@router.post("/audio/upload", response_model=AudioUploadResponse)
async def upload_audio(
    req: AudioUploadRequest,
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    """Upload merged/custom audio from frontend."""
    _require_app_token(x_app_token)

    try:
        # Decode base64 and save to disk
        audio_bytes = base64.b64decode(req.audio_base64)
        os.makedirs(settings.audio_storage_dir, exist_ok=True)

        filename = f"{req.session_id}_uploaded.mp3"
        file_path = os.path.join(settings.audio_storage_dir, filename)
        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        title = req.title or f"Upload: {datetime.date.today().isoformat()}"

        record = insert_audio_narration(
            user_uid=req.user_uid,
            conversation_id=req.conversation_id,
            session_id=req.session_id,
            meditation_type=req.meditation_type,
            audio_type="merged",
            file_path=file_path,
            duration_seconds=req.duration_seconds,
            title=title,
            metadata=req.metadata,
        )

        return {
            "id": record["id"],
            "session_id": req.session_id,
            "created_at": record["created_at"],
        }

    except Exception as exc:
        logger.exception("audio upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="audio_upload_failed")


@router.get("/audio/list", response_model=AudioListResponse)
async def list_audio(
    user_uid: str = Query(...),
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    """List user's audio narrations (max 25, newest first)."""
    _require_app_token(x_app_token)

    rows = list_audio_narrations(user_uid, limit=25)
    narrations = []
    for r in rows:
        filename = os.path.basename(r["file_path"])
        narrations.append(
            AudioNarrationItem(
                id=str(r["id"]),
                session_id=str(r["session_id"]),
                conversation_id=str(r["conversation_id"]),
                meditation_type=r["meditation_type"],
                audio_type=r["audio_type"],
                audio_url=f"{settings.audio_base_url}/stream/{filename}",
                duration_seconds=r.get("duration_seconds"),
                title=r.get("title"),
                metadata=r.get("metadata"),
                created_at=str(r["created_at"]),
            )
        )
    return {"narrations": narrations}


@router.get("/audio/stream/{filename}")
async def stream_audio(filename: str):
    """Stream an audio file for iOS AVPlayer playback."""
    # Sanitize filename to prevent directory traversal
    safe_name = os.path.basename(filename)
    path = os.path.join(settings.audio_storage_dir, safe_name)

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="audio_not_found")

    return FileResponse(path, media_type="audio/mpeg", filename=safe_name)


@router.delete("/audio/{narration_id}")
async def delete_audio(
    narration_id: str,
    user_uid: str = Query(...),
    x_app_token: Optional[str] = Header(default=None),
) -> dict:
    """Delete an audio narration and its file."""
    _require_app_token(x_app_token)

    file_path = delete_audio_narration(narration_id, user_uid)
    if file_path is None:
        raise HTTPException(status_code=404, detail="narration_not_found")

    # Remove file from disk
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        logger.warning("Failed to remove file: %s", file_path)

    return {"ok": True}
