# Neuroheart TTS API

Base URLs:

- Local dev (uvicorn --reload):  `http://localhost:8000`
- On the VPS (docker compose):   `http://127.0.0.1:8844`   (loopback only — no public exposure)
- Behind a reverse proxy:         `https://tts.yourdomain.com`

Content type for all POST bodies: `application/json`.
Response for all successful audio calls: `audio/mpeg`.

Auth: none at the HTTP layer. Add basic-auth / API key / mTLS at the reverse
proxy before exposing publicly.

---

## `GET /health`

Liveness + state.

```bash
curl -s http://localhost:8000/health
# {"status":"ok","clone_ref":"73ca...mp3","uptime_s":42.3}
```

---

## `POST /tts`  (Phase 1 — live)

Synthesize text using the cloned voice (ElevenLabs v3). ElevenLabs natively
honors `<break time="Xs"/>` tags inside the text — pass your SSML verbatim.

### Request body

All fields optional except `text`. Defaults come from `secrets.env`
(`ELEVEN_DEFAULT_*`). Per-request values override the defaults.

| Field                      | Type    | Default           | Notes |
|----------------------------|---------|-------------------|-------|
| `text`                     | string  | **required**      | Speech content. Supports `<break time="Xs"/>`. |
| `model`                    | string  | `eleven_v3`       | `eleven_v3` or `eleven_multilingual_v2`. |
| `stability`                | float   | `0.5`             | 0.0–1.0. Higher = more consistent, less expressive. |
| `similarity_boost`         | float   | `0.75`            | 0.0–1.0. How closely to match the reference voice. |
| `speed`                    | float   | `0.75`            | 0.7–1.3. Playback rate; <1 = slower. |
| `apply_text_normalization` | string  | `auto`            | `auto` / `on` / `off`. |
| `language_code`            | string  | `""`              | ISO-639 hint, empty = auto-detect. |
| `output_format`            | string  | `mp3_44100_192`   | `mp3_44100_192` or `opus_48000_192`. |
| `seed`                     | int     | randomized        | 0–2147483647 for determinism. |
| `remove_background_noise`  | bool    | `false`           | Clean the voice-clone reference before cloning. |

### Example

```bash
curl -X POST http://localhost:8000/tts \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Sit quietly and close your eyes. <break time=\"2s\"/> Take a slow breath in.",
    "speed": 0.75,
    "stability": 0.5,
    "similarity_boost": 0.75
  }' \
  -o outputs/tts.mp3
```

### Errors

| Status | When |
|--------|------|
| 400    | empty `text` |
| 502    | Comfy Cloud / ElevenLabs failure (see body `.detail` for the underlying message) |
| 422    | Pydantic validation failed (out-of-range float, wrong type) |

---

## `POST /music`  (Phase 2 — 501 Not Implemented)

Stub. Accepts the real request shape so clients can be built ahead of the
backend. When enabled, returns an `audio/mpeg` music clip.

### Request body

```json
{
  "prompt": "slow ambient meditation, piano, drone, 60 bpm",
  "duration_seconds": 60,
  "seed": null
}
```

Returns `501 {"error": "not_implemented", "phase": 2, "accepted_params": ...}`.

See `music_chatterbox_implementation_plan.md`.

---

## `POST /narration`  (Phase 2 — 501 Not Implemented)

Stub. Eventually: TTS + looped / ducked music bed → single mp3.

### Request body

```json
{
  "tts":   { "text": "...", "speed": 0.75 },
  "music": { "prompt": "soft ambient piano", "duration_seconds": 60 },
  "music_gain_db": -18
}
```

Returns `501 {"error": "not_implemented", "phase": 2, ...}`.

See `music_chatterbox_implementation_plan.md`.

---

## Operational notes

- **Timeouts**: TTS can take 30–90s on Comfy Cloud. The server has `--timeout-keep-alive 900`;
  if you front it with nginx/traefik, raise `proxy_read_timeout` to at least 900s.
- **Voice reference**: `voice_to_clone.mp3` is uploaded to Comfy Cloud once at startup
  (see the `lifespan` hook in `app/main.py`). To change the voice, replace the file
  and restart the container.
- **Concurrency**: Comfy Cloud serializes jobs per account on the basic tier, so
  simultaneous `/tts` calls queue server-side. This is not a bug in our API.
- **Outputs folder**: only the test scripts persist mp3s there. Production calls
  stream the audio back in the HTTP response — nothing is written to disk on the server.
