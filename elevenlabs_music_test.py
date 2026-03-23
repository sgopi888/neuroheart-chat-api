"""ElevenLabs Music Generation Integration Test.

Run: python elevenlabs_music_test.py

Tests that the ElevenLabs SDK can generate music and save a playable MP3.
Requires ELEVENLABS_API in .env.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API = os.getenv("ELEVENLABS_API", "")

if not ELEVENLABS_API:
    print("ERROR: ELEVENLABS_API not set in .env")
    sys.exit(1)


def test_compose_basic():
    """Test basic music composition — 15 seconds of calm meditation music."""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API)

    print("Generating 15s of calm meditation music...")
    audio_stream = client.music.compose(
        prompt="Calm meditation background music, soft ambient pads, peaceful atmosphere",
        music_length_ms=15000,
    )

    # compose() returns a generator — collect all chunks
    audio_bytes = b"".join(audio_stream)

    assert audio_bytes is not None, "No audio returned"
    assert len(audio_bytes) > 1000, f"Audio too small: {len(audio_bytes)} bytes"

    out_path = "test_music_output.mp3"
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    assert os.path.exists(out_path)
    file_size = os.path.getsize(out_path)
    print(f"SUCCESS: Saved {file_size:,} bytes to {out_path}")

    # Cleanup
    os.remove(out_path)
    print("Cleaned up test file.")


def test_compose_with_mood_context():
    """Test music composition with mood/style context (simulating frontend config)."""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API)

    prompt = (
        "Calm meditation background music, "
        "gentle Indian classical style, "
        "soft sitar and flute, slow tempo, "
        "peaceful and grounding atmosphere"
    )

    print(f"Generating music with mood context: {prompt[:80]}...")
    audio_stream = client.music.compose(
        prompt=prompt,
        music_length_ms=15000,
    )

    audio_bytes = b"".join(audio_stream)

    assert audio_bytes is not None, "No audio returned"
    assert len(audio_bytes) > 1000, f"Audio too small: {len(audio_bytes)} bytes"

    out_path = "test_music_mood_output.mp3"
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    file_size = os.path.getsize(out_path)
    print(f"SUCCESS: Saved {file_size:,} bytes to {out_path}")

    # Cleanup
    os.remove(out_path)
    print("Cleaned up test file.")


def test_compose_60s():
    """Test 60-second music generation (production length for looping)."""
    from elevenlabs import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API)

    print("Generating 60s of meditation music (production length)...")
    audio_stream = client.music.compose(
        prompt="Calm meditation background music, ambient atmosphere, soft pads",
        music_length_ms=60000,
    )

    audio_bytes = b"".join(audio_stream)

    assert audio_bytes is not None, "No audio returned"
    assert len(audio_bytes) > 10000, f"Audio too small for 60s: {len(audio_bytes)} bytes"

    out_path = "test_music_60s_output.mp3"
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    file_size = os.path.getsize(out_path)
    print(f"SUCCESS: 60s music = {file_size:,} bytes saved to {out_path}")

    # Cleanup
    os.remove(out_path)
    print("Cleaned up test file.")


if __name__ == "__main__":
    print("=" * 60)
    print("ElevenLabs Music Generation Integration Test")
    print("=" * 60)

    tests = [
        ("Basic compose (15s)", test_compose_basic),
        ("Compose with mood context (15s)", test_compose_with_mood_context),
        ("Production length compose (60s)", test_compose_60s),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
