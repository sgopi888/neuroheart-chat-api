from elevenlabs import ElevenLabs
from dotenv import load_dotenv
import os

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API"))
audio_stream = client.music.compose(
    prompt="Calm meditation music, soft sitar, peaceful",
    music_length_ms=15000
)

audio_bytes = b"".join(audio_stream)

with open("output.mp3", "wb") as f:
    f.write(audio_bytes)

print("SUCCESS: file generated")