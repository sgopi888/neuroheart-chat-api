"""Meditation audio generation service.

Orchestrates SSML script generation, voice narration (HuggingFace),
ambient music (ElevenLabs), and server-side merging (pydub).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
from typing import Dict, Optional

from app.chat_service import chat_once
from app.config import settings
from app.history_repository import (
    enforce_audio_limit,
    insert_audio_narration,
)
from app.openai_client import call_gpt_mem0
from app.prompts import (
    MEDITATION_GENERATION_LONG_PROMPT,
    MEDITATION_GENERATION_MEDIUM_PROMPT,
    MEDITATION_GENERATION_SHORT_PROMPT,
    MEDITATION_TITLE_PROMPT,
)

logger = logging.getLogger(__name__)


def _meditation_type(duration: int) -> str:
    """Map duration (minutes) to meditation type."""
    if duration <= 2:
        return "short"
    if duration <= 15:
        return "medium"
    return "deep"


def _build_music_prompt(music_config: Optional[Dict]) -> str:
    """Build an ElevenLabs music prompt from the frontend-supplied config."""
    parts = ["Calm meditation background music"]
    if music_config:
        if music_config.get("mood"):
            parts.append(music_config["mood"])
        if music_config.get("style"):
            parts.append(music_config["style"])
        if music_config.get("additional_context"):
            parts.append(music_config["additional_context"])
    return ", ".join(parts)


async def _generate_ssml_script(
    user_uid: str,
    conversation_id: str,
    mood: str,
    depth: str | None,
    duration: int,
    session_type: str,
    meditation_type: str,
) -> str:
    """Generate an SSML meditation script using GPT with full context."""
    prompts = {
        "short": MEDITATION_GENERATION_SHORT_PROMPT,
        "medium": MEDITATION_GENERATION_MEDIUM_PROMPT,
        "deep": MEDITATION_GENERATION_LONG_PROMPT,
    }
    prompt_template = prompts[meditation_type]

    # The prompts are self-contained — just send as the user message
    # chat_once will add HRV, memories, RAG, summaries as context
    response = await chat_once(
        user_uid=user_uid,
        conversation_id=conversation_id,
        user_message=prompt_template,
        hrv_range="7d",
    )
    return response["reply"]


async def _generate_title(script: str, mood: str) -> str:
    """Ask LLM for a unique 3-5 word title based on the script and mood."""
    excerpt = script[:500]
    prompt = MEDITATION_TITLE_PROMPT.format(mood=mood, excerpt=excerpt)
    try:
        title = await asyncio.to_thread(
            call_gpt_mem0,
            [{"role": "user", "content": prompt}],
        )
        # Clean up: strip quotes, limit length
        title = title.strip().strip('"').strip("'")
        if len(title) > 60:
            title = title[:60]
        return title
    except Exception:
        logger.exception("Title generation failed, using fallback")
        import datetime
        return f"Meditation: {mood} {datetime.date.today().isoformat()}"


async def _generate_voice(script_text: str, session_id: str) -> str | None:
    """Generate voice narration via HuggingFace Gradio space.

    Returns the file path to the saved MP3, or None on failure.
    """
    if not settings.hf_token or not settings.hf_space:
        logger.warning("HF_TOKEN or HF_SPACE not configured, skipping voice generation")
        return None

    def _call_hf():
        from gradio_client import Client

        client = Client(settings.hf_space, token=settings.hf_token)
        result = client.predict(script_text=script_text, voice_name="Drew", api_name="/predict")
        return result

    try:
        result = await asyncio.to_thread(_call_hf)

        # result may be a file path string or a dict with url
        if isinstance(result, str):
            src_path = result
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            item = result[0]
            if isinstance(item, dict) and "url" in item:
                # Download from URL
                import httpx

                resp = httpx.get(
                    item["url"],
                    headers={"Authorization": f"Bearer {settings.hf_token}"},
                    timeout=120,
                )
                resp.raise_for_status()
                dest = os.path.join(
                    settings.audio_storage_dir, f"{session_id}_voice.mp3"
                )
                with open(dest, "wb") as f:
                    f.write(resp.content)
                logger.info("Voice audio saved: %s (%d bytes)", dest, len(resp.content))
                return dest
            else:
                src_path = str(item)
        else:
            src_path = str(result)

        # Copy local file to storage dir
        dest = os.path.join(settings.audio_storage_dir, f"{session_id}_voice.mp3")
        shutil.copy2(src_path, dest)
        logger.info("Voice audio saved: %s", dest)
        return dest

    except Exception:
        logger.exception("Voice generation failed")
        return None


async def _generate_music(
    music_config: Optional[Dict], session_id: str
) -> str | None:
    """Generate ambient music via ElevenLabs SDK.

    Generates 1 minute of music. Returns file path or None on failure.
    """
    if not settings.elevenlabs_api_key:
        logger.warning("ELEVENLABS_API not configured, skipping music generation")
        return None

    music_prompt = _build_music_prompt(music_config)

    def _call_elevenlabs():
        from elevenlabs import ElevenLabs

        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        # compose() returns a generator (stream) — collect all chunks
        audio_stream = client.music.compose(
            prompt=music_prompt,
            music_length_ms=60000,  # 60 seconds in milliseconds
        )
        return b"".join(audio_stream)

    try:
        audio_bytes = await asyncio.to_thread(_call_elevenlabs)

        dest = os.path.join(settings.audio_storage_dir, f"{session_id}_music.mp3")
        with open(dest, "wb") as f:
            f.write(audio_bytes)
        logger.info("Music audio saved: %s (%d bytes)", dest, len(audio_bytes))
        return dest

    except Exception:
        logger.exception("Music generation failed")
        return None


def _merge_audio(voice_path: str, music_path: str, session_id: str) -> str:
    """Merge voice + music: loop music to voice length, mix at 15% volume.

    Returns path to merged MP3.
    """
    from pydub import AudioSegment

    voice = AudioSegment.from_file(voice_path)
    music = AudioSegment.from_file(music_path)

    # Loop music to match voice length
    if len(music) > 0:
        loops_needed = (len(voice) // len(music)) + 1
        music_looped = (music * loops_needed)[: len(voice)]
    else:
        music_looped = music

    # Mix: voice at 100%, music at ~15% (-16.5 dB reduction)
    music_quiet = music_looped - 16.5
    merged = voice.overlay(music_quiet)

    dest = os.path.join(settings.audio_storage_dir, f"{session_id}_merged.mp3")
    merged.export(dest, format="mp3")
    logger.info("Merged audio saved: %s (%d ms)", dest, len(merged))
    return dest


async def generate_meditation(
    user_uid: str,
    conversation_id: str,
    mood: str,
    depth: str | None,
    duration: int,
    session_type: str,
    music_config: Optional[Dict] = None,
) -> Dict:
    """Full meditation generation pipeline.

    1. Generate SSML script via GPT
    2. Generate voice + music in parallel
    3. Merge server-side
    4. Save to DB, enforce max 25
    5. Return session_id, script, audio_url, meditation_type
    """
    med_type = _meditation_type(duration)
    session_id = str(uuid.uuid4())

    # Ensure storage directory exists
    os.makedirs(settings.audio_storage_dir, exist_ok=True)

    # Step 1: Generate SSML script
    logger.info(
        "Generating %s meditation for user=%s session=%s",
        med_type, user_uid, session_id,
    )
    script = await _generate_ssml_script(
        user_uid, conversation_id, mood, depth, duration, session_type, med_type
    )

    # Step 2: Generate voice + music in parallel
    voice_task = _generate_voice(script, session_id)

    if music_config and music_config.get("enabled") is False:
        music_task = None
    else:
        music_task = _generate_music(music_config, session_id)

    if music_task:
        voice_result, music_result = await asyncio.gather(
            voice_task, music_task, return_exceptions=True
        )
    else:
        voice_result = await voice_task
        music_result = None

    # Handle exceptions from gather
    voice_path = voice_result if isinstance(voice_result, str) else None
    music_path = music_result if isinstance(music_result, str) else None

    if isinstance(voice_result, Exception):
        logger.error("Voice generation raised: %s", voice_result)
    if isinstance(music_result, Exception):
        logger.error("Music generation raised: %s", music_result)

    if not voice_path:
        raise RuntimeError("Voice generation failed — cannot produce meditation audio")

    # Step 3: Merge if music available, otherwise use voice-only
    if music_path:
        try:
            merged_path = await asyncio.to_thread(
                _merge_audio, voice_path, music_path, session_id
            )
        except Exception:
            logger.exception("Merge failed, falling back to voice-only")
            merged_path = voice_path
    else:
        logger.warning("No music generated, using voice-only audio")
        merged_path = voice_path

    # Calculate duration from merged file
    duration_seconds = None
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(merged_path)
        duration_seconds = int(len(audio) / 1000)
    except Exception:
        pass

    # Step 4: Generate unique title via LLM
    title = await _generate_title(script, mood)

    insert_audio_narration(
        user_uid=user_uid,
        conversation_id=conversation_id,
        session_id=session_id,
        meditation_type=med_type,
        audio_type="merged",
        file_path=merged_path,
        duration_seconds=duration_seconds,
        title=title,
        metadata={
            "mood": mood,
            "depth": depth,
            "duration_minutes": duration,
            "session_type": session_type,
            "music_config": music_config,
            "has_music": music_path is not None,
        },
    )

    # Step 5: Enforce max 25 limit, clean up old files
    deleted_paths = enforce_audio_limit(user_uid, max_count=25)
    for p in deleted_paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            logger.warning("Failed to remove old audio file: %s", p)

    # Build streaming URL
    filename = os.path.basename(merged_path)
    import time
    audio_url = f"{settings.audio_base_url}/stream/{filename}?ts={int(time.time())}"

    return {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "script": script,
        "title": title,
        "audio_url": audio_url,
        "meditation_type": med_type,
    }
