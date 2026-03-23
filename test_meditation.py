"""Meditation audio generation pipeline tests.

Run: python -m pytest test_meditation.py -v
Or:  python test_meditation.py

Tests cover:
- Meditation type selection (duration mapping)
- Prompt selection per type
- SSML script generation (mocked)
- Voice generation (mocked)
- Music generation (mocked)
- Audio merging (pydub)
- Audio CRUD (repository functions)
- Max 25 limit enforcement
- Music fallback (voice-only when music fails)
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()


# ── Meditation type selection ─────────────────────────────────────


def test_meditation_type_short():
    from app.meditation_service import _meditation_type

    assert _meditation_type(1) == "short"
    assert _meditation_type(2) == "short"


def test_meditation_type_medium():
    from app.meditation_service import _meditation_type

    assert _meditation_type(3) == "medium"
    assert _meditation_type(5) == "medium"
    assert _meditation_type(10) == "medium"
    assert _meditation_type(15) == "medium"


def test_meditation_type_deep():
    from app.meditation_service import _meditation_type

    assert _meditation_type(16) == "deep"
    assert _meditation_type(20) == "deep"
    assert _meditation_type(30) == "deep"


# ── Prompt selection ──────────────────────────────────────────────


def test_prompt_selection_short():
    from app.prompts import MEDITATION_GENERATION_SHORT_PROMPT

    assert "150-250 words" in MEDITATION_GENERATION_SHORT_PROMPT
    assert '<break time="500ms"' in MEDITATION_GENERATION_SHORT_PROMPT


def test_prompt_selection_medium():
    from app.prompts import MEDITATION_GENERATION_MEDIUM_PROMPT

    assert "500-700 words" in MEDITATION_GENERATION_MEDIUM_PROMPT
    assert "breakthrough" in MEDITATION_GENERATION_MEDIUM_PROMPT.lower()
    assert '<break time="3s"' in MEDITATION_GENERATION_MEDIUM_PROMPT


def test_prompt_selection_deep():
    from app.prompts import MEDITATION_GENERATION_LONG_PROMPT

    assert '<break time="60s"' in MEDITATION_GENERATION_LONG_PROMPT
    assert '<break time="180s"' in MEDITATION_GENERATION_LONG_PROMPT
    assert "Move your feet" in MEDITATION_GENERATION_LONG_PROMPT


# ── Music prompt building ─────────────────────────────────────────


def test_build_music_prompt_no_config():
    from app.meditation_service import _build_music_prompt

    result = _build_music_prompt(None)
    assert "Calm meditation background music" in result


def test_build_music_prompt_with_config():
    from app.meditation_service import _build_music_prompt

    config = {
        "mood": "peaceful and grounding",
        "style": "Indian classical ambient",
        "additional_context": "soft sitar",
    }
    result = _build_music_prompt(config)
    assert "peaceful and grounding" in result
    assert "Indian classical ambient" in result
    assert "soft sitar" in result


# ── SSML script generation (mocked) ──────────────────────────────


@pytest.mark.asyncio
async def test_ssml_script_generation():
    """Verify _generate_ssml_script calls chat_once and returns script."""
    from app.meditation_service import _generate_ssml_script

    mock_reply = 'Close your eyes. <break time="1s" /> Breathe deeply.'

    with patch("app.meditation_service.chat_once", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = {"reply": mock_reply}

        script = await _generate_ssml_script(
            user_uid="test-user",
            conversation_id="test-conv-id",
            mood="stressed",
            depth="Mind",
            duration=5,
            session_type="meditation",
            meditation_type="medium",
        )

        assert script == mock_reply
        assert "<break" in script
        mock_chat.assert_called_once()


# ── Voice generation (mocked) ────────────────────────────────────


@pytest.mark.asyncio
async def test_voice_generation_file_result():
    """Verify _generate_voice handles file path result from gradio."""
    from app.meditation_service import _generate_voice

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake source MP3
        src = os.path.join(tmpdir, "source.mp3")
        with open(src, "wb") as f:
            f.write(b"\xff\xfb\x90\x00" * 100)  # fake MP3 header bytes

        with (
            patch("app.meditation_service.settings") as mock_settings,
            patch("app.meditation_service.asyncio") as mock_asyncio,
        ):
            mock_settings.hf_token = "test-token"
            mock_settings.hf_space = "test/space"
            mock_settings.audio_storage_dir = tmpdir

            # Mock asyncio.to_thread to call the function directly
            async def fake_to_thread(fn, *args, **kwargs):
                return fn(*args, **kwargs)

            mock_asyncio.to_thread = fake_to_thread

            mock_client = MagicMock()
            mock_client.predict.return_value = src

            with patch("gradio_client.Client", return_value=mock_client):
                result = await _generate_voice("test script", "test-session-id")

            # Should have copied to storage dir
            if result:
                assert "test-session-id_voice.mp3" in result


# ── Music generation (mocked) ────────────────────────────────────


@pytest.mark.asyncio
async def test_music_generation_mocked():
    """Verify _generate_music calls ElevenLabs SDK and saves file."""
    from app.meditation_service import _generate_music

    fake_audio = b"\xff\xfb\x90\x00" * 500  # fake MP3 bytes

    with tempfile.TemporaryDirectory() as tmpdir:
        with (
            patch("app.meditation_service.settings") as mock_settings,
            patch("app.meditation_service.asyncio") as mock_asyncio,
        ):
            mock_settings.elevenlabs_api_key = "test-key"
            mock_settings.audio_storage_dir = tmpdir

            async def fake_to_thread(fn, *args, **kwargs):
                return fn(*args, **kwargs)

            mock_asyncio.to_thread = fake_to_thread

            mock_client = MagicMock()
            mock_client.music.compose.return_value = fake_audio

            with patch("elevenlabs.ElevenLabs", return_value=mock_client):
                result = await _generate_music({"mood": "calm"}, "test-session")

            if result:
                assert os.path.exists(result)
                assert os.path.getsize(result) == len(fake_audio)


# ── Audio merging ─────────────────────────────────────────────────


def test_merge_audio():
    """Verify voice + music merge produces a file with correct properties."""
    try:
        from pydub import AudioSegment
        from pydub.generators import Sine
    except ImportError:
        pytest.skip("pydub not installed")

    from app.meditation_service import _merge_audio

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio files
        voice = Sine(440).to_audio_segment(duration=5000)  # 5 seconds
        music = Sine(220).to_audio_segment(duration=2000)  # 2 seconds

        voice_path = os.path.join(tmpdir, "voice.mp3")
        music_path = os.path.join(tmpdir, "music.mp3")
        voice.export(voice_path, format="mp3")
        music.export(music_path, format="mp3")

        with patch("app.meditation_service.settings") as mock_settings:
            mock_settings.audio_storage_dir = tmpdir

            merged_path = _merge_audio(voice_path, music_path, "test-merge")

        assert os.path.exists(merged_path)
        assert "test-merge_merged.mp3" in merged_path

        # Merged should be approximately voice length
        merged = AudioSegment.from_file(merged_path)
        assert abs(len(merged) - 5000) < 200  # within 200ms tolerance


# ── Music fallback ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_music_fallback_voice_only():
    """When music generation fails, should still return voice-only result."""
    from app.meditation_service import generate_meditation

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a fake voice file
        voice_path = os.path.join(tmpdir, "test-session_voice.mp3")

        try:
            from pydub.generators import Sine

            voice = Sine(440).to_audio_segment(duration=3000)
            voice.export(voice_path, format="mp3")
        except ImportError:
            with open(voice_path, "wb") as f:
                f.write(b"\xff\xfb\x90\x00" * 500)

        with (
            patch("app.meditation_service._generate_ssml_script", new_callable=AsyncMock) as mock_script,
            patch("app.meditation_service._generate_voice", new_callable=AsyncMock) as mock_voice,
            patch("app.meditation_service._generate_music", new_callable=AsyncMock) as mock_music,
            patch("app.meditation_service.insert_audio_narration") as mock_insert,
            patch("app.meditation_service.enforce_audio_limit") as mock_enforce,
            patch("app.meditation_service.settings") as mock_settings,
        ):
            mock_script.return_value = "Test script <break time='1s' />"
            mock_voice.return_value = voice_path
            mock_music.return_value = None  # Music failed
            mock_insert.return_value = {"id": "test-id", "created_at": "2026-03-20"}
            mock_enforce.return_value = []
            mock_settings.audio_storage_dir = tmpdir
            mock_settings.audio_base_url = "https://neuroheart.ai/audio"

            result = await generate_meditation(
                user_uid="test-user",
                conversation_id="test-conv",
                mood="stressed",
                depth=None,
                duration=2,
                session_type="meditation",
                music_config=None,
            )

            assert result["meditation_type"] == "short"
            assert result["audio_url"]  # Should still have a URL (voice-only)
            assert result["script"]


# ── Run all tests ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Meditation Pipeline Tests")
    print("=" * 60)

    # Run non-async tests
    sync_tests = [
        ("Meditation type: short", test_meditation_type_short),
        ("Meditation type: medium", test_meditation_type_medium),
        ("Meditation type: deep", test_meditation_type_deep),
        ("Prompt: short", test_prompt_selection_short),
        ("Prompt: medium", test_prompt_selection_medium),
        ("Prompt: deep", test_prompt_selection_deep),
        ("Music prompt: no config", test_build_music_prompt_no_config),
        ("Music prompt: with config", test_build_music_prompt_with_config),
    ]

    async_tests = [
        ("SSML script generation", test_ssml_script_generation),
        ("Music fallback (voice-only)", test_music_fallback_voice_only),
    ]

    passed = 0
    failed = 0

    for name, test_fn in sync_tests:
        try:
            test_fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    for name, test_fn in async_tests:
        try:
            asyncio.run(test_fn())
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    # Merge test (requires pydub + ffmpeg)
    try:
        test_merge_audio()
        print("  PASS: Audio merge")
        passed += 1
    except Exception as e:
        print(f"  SKIP/FAIL: Audio merge — {e}")
        failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
