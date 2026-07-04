"""Meditation audio generation endpoints."""

from __future__ import annotations

import base64
import datetime
import logging
import os
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from fastapi.responses import FileResponse

from app.auth import assert_user_scope, get_verified_user_uid, require_app_token
from app.config import settings
from app.history_repository import (
    delete_audio_narration,
    insert_audio_narration,
    insert_message,
    list_audio_narrations,
)
from app.meditation_jobs_repository import (
    complete_job,
    create_job,
    fail_job,
    get_job,
    mark_running,
)
from app.meditation_service import generate_meditation
from app.schemas import (
    AudioListResponse,
    AudioNarrationItem,
    AudioUploadRequest,
    AudioUploadResponse,
    GenerateMeditationJobResponse,
    GenerateMeditationRequest,
    GenerateMeditationResponse,
    MeditationJobStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/practice", tags=["meditation"])


@router.post("/generate-meditation", response_model=GenerateMeditationResponse)
async def generate_meditation_endpoint(
    req: GenerateMeditationRequest,
    verified_user_uid: str = Depends(get_verified_user_uid),
) -> dict:
    """Generate SSML meditation script, voice narration, ambient music, and merge."""
    assert_user_scope(verified_user_uid, req.user_uid)

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

        # Persist a meditation link message in the conversation so it
        # survives history reloads on the client.
        if result.get("audio_url"):
            title = result.get("title", "Meditation")
            med_message = (
                f"🧘 Your meditation is ready: \"{title}\"\n\n"
                f"Tap to play in the Practice tab.\n"
                f"[MEDITATION_AUDIO:{result['audio_url']}:{title}]"
            )
            try:
                insert_message(
                    user_uid=req.user_uid,
                    conversation_id=req.conversation_id,
                    role="assistant",
                    content=med_message,
                    metadata={
                        "type": "meditation_link",
                        "session_id": result.get("session_id"),
                        "audio_url": result["audio_url"],
                        "title": title,
                    },
                )
            except Exception:
                logger.warning("Could not save meditation message to conversation")

        return result

    except LookupError:
        raise HTTPException(status_code=404, detail="conversation_not_found")
    except RuntimeError as exc:
        logger.error("Meditation generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("generate_meditation failed: %s", exc)
        raise HTTPException(status_code=500, detail="meditation_generation_failed")


# --- Async meditation generation (job + poll) ---------------------------------
#
# The synchronous endpoint above blocks 1-2 minutes, so if the phone is locked or
# the app is backgrounded mid-request the result is lost. These two endpoints let
# the client start generation, get a job_id immediately, and poll for the result,
# which survives lock / app-kill / reboot because the work runs server-side.


async def _run_meditation_job(
    job_id: str,
    user_uid: str,
    conversation_id: str,
    mood: str,
    depth: Optional[str],
    duration: int,
    session_type: str,
    music_config: Optional[dict],
) -> None:
    """Background worker: runs the same generation as the sync endpoint, then
    records the outcome on the job row. Reuses generate_meditation(), which also
    persists the meditation_link chat message + audio narration on success."""
    mark_running(job_id)
    try:
        result = await generate_meditation(
            user_uid=user_uid,
            conversation_id=conversation_id,
            mood=mood,
            depth=depth,
            duration=duration,
            session_type=session_type,
            music_config=music_config,
        )

        audio_url = result.get("audio_url")
        title = result.get("title", "Meditation")

        # Mirror the sync endpoint: persist a meditation link message so it shows
        # up on history reload even if the client never polls.
        if audio_url:
            med_message = (
                f"🧘 Your meditation is ready: \"{title}\"\n\n"
                f"Tap to play in the Practice tab.\n"
                f"[MEDITATION_AUDIO:{audio_url}:{title}]"
            )
            try:
                insert_message(
                    user_uid=user_uid,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=med_message,
                    metadata={
                        "type": "meditation_link",
                        "session_id": result.get("session_id"),
                        "audio_url": audio_url,
                        "title": title,
                    },
                )
            except Exception:
                logger.warning("Could not save meditation message to conversation")

        complete_job(
            job_id=job_id,
            session_id=result.get("session_id"),
            title=title,
            audio_url=audio_url,
            meditation_type=result.get("meditation_type"),
        )
    except Exception as exc:
        logger.exception("async meditation job %s failed: %s", job_id, exc)
        fail_job(job_id, str(exc))


@router.post("/generate-meditation-async", response_model=GenerateMeditationJobResponse)
async def generate_meditation_async(
    req: GenerateMeditationRequest,
    background_tasks: BackgroundTasks,
    verified_user_uid: str = Depends(get_verified_user_uid),
) -> dict:
    """Start meditation generation and return a job_id immediately."""
    assert_user_scope(verified_user_uid, req.user_uid)

    music_config = None
    if req.music_config:
        music_config = req.music_config.model_dump(exclude_none=True)

    job_id = create_job(
        user_uid=req.user_uid,
        conversation_id=req.conversation_id,
        mood=req.mood,
        depth=req.depth,
        duration=req.duration,
        session_type=req.session_type,
        music_config=music_config,
    )

    background_tasks.add_task(
        _run_meditation_job,
        job_id,
        req.user_uid,
        req.conversation_id,
        req.mood,
        req.depth,
        req.duration,
        req.session_type,
        music_config,
    )

    return {"job_id": job_id, "status": "pending"}


@router.get("/job/{job_id}", response_model=MeditationJobStatusResponse)
async def get_meditation_job(
    job_id: str,
    user_uid: str = Query(...),
    verified_user_uid: str = Depends(get_verified_user_uid),
) -> dict:
    """Poll the status/result of an async meditation job."""
    assert_user_scope(verified_user_uid, user_uid)

    job = get_job(job_id, user_uid)
    if job is None:
        raise HTTPException(status_code=404, detail="job_not_found")

    return {
        "job_id": str(job["id"]),
        "status": job["status"],
        "session_id": str(job["session_id"]) if job.get("session_id") else None,
        "title": job.get("title"),
        "audio_url": job.get("audio_url"),
        "meditation_type": job.get("meditation_type"),
        "error": job.get("error"),
    }


@router.post("/audio/upload", response_model=AudioUploadResponse)
async def upload_audio(
    req: AudioUploadRequest,
    verified_user_uid: str = Depends(get_verified_user_uid),
) -> dict:
    """Upload merged/custom audio from frontend."""
    assert_user_scope(verified_user_uid, req.user_uid)

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
    verified_user_uid: str = Depends(get_verified_user_uid),
) -> dict:
    """List user's audio narrations (max 25, newest first)."""
    assert_user_scope(verified_user_uid, user_uid)

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
                audio_url=f"{settings.audio_base_url}/stream/{filename}?app_token={settings.app_token}",
                duration_seconds=r.get("duration_seconds"),
                title=r.get("title"),
                metadata=r.get("metadata"),
                created_at=r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else str(r["created_at"]),
            )
        )
    return {"narrations": narrations}


@router.get("/audio/stream/{filename}")
async def stream_audio(
    filename: str,
    app_token: Optional[str] = Query(default=None),
    x_app_token: Optional[str] = Header(default=None),
):
    """Stream an audio file for iOS AVPlayer playback."""
    require_app_token(x_app_token or app_token)
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
    verified_user_uid: str = Depends(get_verified_user_uid),
) -> dict:
    """Delete an audio narration and its file."""
    assert_user_scope(verified_user_uid, user_uid)

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
