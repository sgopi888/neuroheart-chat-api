# NeuroHeart Chat API — Systems Architecture

A FastAPI microservice that powers the NeuroHeart iOS health-coaching app. It fuses heart-rate-variability (HRV) signals, retrieval-augmented generation (RAG), multi-layer memory, calendar context, and generative meditation audio into a single conversational AI surface.

- **Repo:** `/Users/sreekanthgopi/Desktop/mobileapps/neuroheart-chat-api`
- **Port:** 8003
- **Host:** `neuroheart.ai` (VPS 159.198.44.98), systemd unit `neuroheart-chat`
- **Reverse proxy:** Nginx → `127.0.0.1:8003`

---

## 1. High-Level System Diagram

```
                    ┌──────────────────────────────────────────────┐
                    │              iOS App (SwiftUI)               │
                    │  HealthKit · EventKit · AVPlayer · Apple ID  │
                    └────────────────────┬─────────────────────────┘
                                         │ HTTPS (x-apple-id-token / x-app-token)
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │         FastAPI App (app/main.py)            │
                    │  Routers: auth · chat · ingest · practice ·  │
                    │  meditation · mindfulness · calendar         │
                    └──┬──────────┬─────────┬──────────┬───────────┘
                       │          │         │          │
            ┌──────────▼┐  ┌──────▼──────┐ ┌▼──────┐ ┌─▼─────────────┐
            │PostgreSQL │  │ Qdrant      │ │OpenAI │ │ External APIs │
            │ neuroheart│  │ documents1  │ │GPT-5- │ │ HF Gradio TTS │
            │ chat/HRV/ │  │ user_memo-  │ │nano + │ │ ElevenLabs    │
            │ calendar/ │  │ ries        │ │embed-3│ │ ComfyUI /narr │
            │ audio meta│  │ (1536-d cos)│ │-small │ │ HRV API :8002 │
            └───────────┘  └─────────────┘ └───────┘ │ Apple JWKS    │
                                                     │ LangSmith     │
                                                     └───────────────┘
```

Cross-cutting: token-bucket rate limiting (in-memory), tiktoken-based token budgeting, asyncio fan-out, optional LangSmith tracing.

---

## 2. Module Map (`app/`)

| File | Role |
|------|------|
| [app/main.py](app/main.py) | FastAPI bootstrap; mounts all routers; `/health`. |
| [app/config.py](app/config.py) | Frozen `Settings` dataclass; env-driven hyperparameters. |
| [app/db.py](app/db.py) | SQLAlchemy engine singleton (`pool_pre_ping`). |
| [app/schemas.py](app/schemas.py) | Pydantic request/response models. |
| [app/auth.py](app/auth.py) | Apple Sign-In JWT verification; JWKS cache. |
| [app/auth_router.py](app/auth_router.py) | `/v1/auth/*`. |
| [app/chat_router.py](app/chat_router.py) | `/v1/chat/*`. |
| [app/chat_service.py](app/chat_service.py) | Core orchestrator (`chat_once`). |
| [app/history_repository.py](app/history_repository.py) | Repository for conversations / messages / summaries / audio. |
| [app/rag_service.py](app/rag_service.py) | Qdrant semantic retrieval. |
| [app/memory_service.py](app/memory_service.py) | Layer-2 long-term memory + cross-chat profile. |
| [app/openai_client.py](app/openai_client.py) | OpenAI client singleton + LangSmith wrap. |
| [app/llm_observability.py](app/llm_observability.py) | `traceable_call`, `wrap_openai_client`. |
| [app/token_budget.py](app/token_budget.py) | tiktoken `cl100k_base` counting/trimming. |
| [app/rate_limit.py](app/rate_limit.py) | In-memory token-bucket. |
| [app/prompts.py](app/prompts.py) | Centralized system / extraction / summarization prompts. |
| [app/hrv_client.py](app/hrv_client.py) | Async client for the remote HRV analysis API. |
| [app/hrv_apple.py](app/hrv_apple.py) | Local HRV from `health_samples` (Apple HealthKit ingest). |
| [app/hrv_neurokit.py](app/hrv_neurokit.py) | NeuroKit2-based HRV (time/freq/nonlinear). |
| [app/hrv_bpm_per_min.py](app/hrv_bpm_per_min.py) | Calm-score pipeline (Baevsky SI, HF/LF, state classification). |
| [app/hrv_utils.py](app/hrv_utils.py) | Pure-Python time-domain HRV from RR. |
| [app/ingest_router.py](app/ingest_router.py) | `/v1/ingest` — HealthKit sample ingestion. |
| [app/practice_router.py](app/practice_router.py) | `/v1/practice/generate` — text breathing scripts. |
| [app/meditation_service.py](app/meditation_service.py) | SSML → voice → music → merge → persist. |
| [app/meditation_router.py](app/meditation_router.py) | `/v1/practice/generate-meditation`, `/audio/*`. |
| [app/mindfulness_router.py](app/mindfulness_router.py) | `/v1/mindfulness/*` pre/post-HRV sessions. |
| [app/calendar_sync.py](app/calendar_sync.py) | `/v1/calendar/sync` + `format_calendar_context`. |

---

## 3. API Surface (consumed by iOS)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET  | `/health` | – | Liveness |
| POST | `/v1/auth/register` | `x-apple-id-token` | Verify Apple JWT, upsert user |
| GET  | `/v1/auth/me` | `x-apple-id-token` | Profile fetch |
| PUT  | `/v1/auth/me` | `x-apple-id-token` | Update name / age_range |
| POST | `/v1/chat/conversations` | `x-app-token` | Create conversation |
| GET  | `/v1/chat/conversations` | `x-app-token` | List conversations |
| GET  | `/v1/chat/history` | `x-app-token` | Paginated message history (`before_id`) |
| POST | `/v1/chat` | `x-app-token` | One chat turn (rate-limited) |
| POST | `/v1/ingest` | `x-app-token` | HealthKit batch ingestion |
| POST | `/v1/practice/generate` | `x-app-token` | Breathing script (text) |
| POST | `/v1/practice/generate-meditation` | `x-app-token` | Full meditation (script + audio) |
| POST | `/v1/practice/audio/upload` | `x-app-token` | User-supplied MP3 (base64) |
| GET  | `/v1/practice/audio/list` | `x-app-token` | 25 most-recent meditations |
| GET  | `/v1/practice/audio/stream/{file}` | `x-app-token` | MP3 streaming for AVPlayer |
| DELETE | `/v1/practice/audio/{narration_id}` | `x-app-token` | Delete + unlink |
| POST | `/v1/mindfulness/session` | `x-app-token` | Pre/post-HRV + RR → calm score |
| GET  | `/v1/mindfulness/sessions` | `x-app-token` | List |
| GET  | `/v1/mindfulness/session/{id}` | `x-app-token` | Detail + snapshots |
| POST | `/v1/calendar/sync` | `x-app-token` | Upsert EventKit events |

---

## 4. End-to-End Chat Pipeline (`chat_once`)

```
POST /v1/chat
  │
  ├─ rate_limit.allow(user_uid)            # token bucket: cap=20, refill=20/60 tps → 429 if exhausted
  │
  └─ chat_service.chat_once()
      ├─ 1. INSERT user message → chat_messages
      │
      ├─ 2. Parallel context fan-out (asyncio.gather + to_thread)
      │     ├─ fetch_history(limit=15+2)                    [PostgreSQL]
      │     ├─ fetch_hrv_context(hrv_range)                 [hrv_apple OR hrv_client]
      │     ├─ retrieve_rag(user_msg, top_k=10)             [Qdrant cosine]
      │     ├─ retrieve_memories(user_uid, query, k=5)      [Qdrant filter user_uid]
      │     └─ format_calendar_context()                     [PostgreSQL]
      │
      ├─ 3. Layer-3 rolling summarization
      │     trigger: msg_count > 50  OR  older_window_tokens > 50k
      │     → call_gpt_mem0(SUMMARIZATION_SYSTEM + template)
      │     → trim_text_to_tokens(800) → UPDATE conversation_summaries
      │
      ├─ 4. _build_prompt()  (strict 100k token budget)
      │     priority order:
      │       1) SYSTEM_PROMPT
      │       2) Layer-3 rolling summary
      │       3) Layer-2 long-term memories
      │       4) Cross-chat profile (30-day window, ≤50 lines)
      │       5) HRV compact block (CSV-like)
      │       6) Calendar block
      │       7) RAG passages (≤300 tok each, ≤20 chunks)
      │       8) Recent history (newest-last)
      │     overflow strategy:
      │       reduce RAG  20 → 10 → 5 → 0
      │       reduce hist 15 → 12 → 8 → 4 → 0
      │
      ├─ 5. call_gpt(messages)                              [@traceable_call]
      │     model = OPENAI_MODEL (gpt-5-nano), max_completion_tokens=16384
      │
      ├─ 6. INSERT assistant message → chat_messages (metadata: hrv_range, rag_k)
      │
      ├─ 7. Background asyncio.create_task fan-out (non-blocking):
      │     ├─ extract_and_store_memories()      Layer-2 facts → Qdrant
      │     ├─ update_cross_chat_profile()       per-day one-liner
      │     └─ if reply has [GENERATE_MEDITATION]: _generate_meditation_background()
      │
      └─ 8. Side-channel signals to iOS:
            calendar_change   = keyword scan ("add to calendar", "scheduled", …)
            generate_meditation = tag detected (stripped from displayed reply)
```

Concurrency: blocking I/O (Qdrant, OpenAI, SQLAlchemy) wrapped in `asyncio.to_thread`; outbound HRV/HF calls are native `httpx.AsyncClient`.

---

## 5. Three-Layer Memory Architecture

| Layer | Storage | Lifetime | Trigger | Used in prompt as |
|---|---|---|---|---|
| **L1 Short-term** | `chat_messages` (Postgres) | Permanent | Every turn | Recent 15 turns verbatim |
| **L2 Long-term semantic** | Qdrant `user_memories` (1536-d, cosine) | Unbounded (cap 200/user) | Background after each turn | Top-5 facts above 0.20 similarity |
| **L2+ Cross-chat profile** | `user_cross_chat_profiles` (Postgres) | 30-day window, ≤50 dated lines | Background after each turn | Compact dated digest |
| **L3 Rolling summary** | `conversation_summaries` (Postgres) | Per-conversation | When `msg_count>50` or older-window > 50k tok | 800-token structured summary |

L2 specifics:
- Extraction prompt → JSON array of facts via `call_gpt_mem0` (cheaper model).
- Embed with `text-embedding-3-small`; dedup if cosine ≥ **0.92**.
- Daily $2 budget tracked in-process; gating on user/assistant message lengths (≥40 / ≥50 chars).

---

## 6. RAG Subsystem

- **Vector store:** Qdrant collection `documents1`, 1536-dim, cosine.
- **Embeddings:** OpenAI `text-embedding-3-small` (singleton client).
- **Filter logic** (multi-tenant): match docs where (no `type`) OR `type="knowledge"` OR (`type="memory"` AND `user_uid=<uid>`).
- **Post-processing:** dedup by first 80 chars of passage, truncate to 600 chars, return `{score, text, source, type}`.
- **Budgeting:** ≤20 chunks × 300 tokens each = ≤6k tokens, gracefully shrunk under pressure.

Algorithmic ideas applied: ANN search (HNSW inside Qdrant), cosine-similarity dedup, type-aware filter union.

---

## 7. HRV Pipeline

### 7.1 Sources & Tiers

| Tier | `health_samples.sample_type` | Payload | Quality |
|---|---|---|---|
| 1 | `heartbeat_series` | `rr_intervals[]` (ms) | Best (full RR series) |
| 2 | `hrv_sdnn` | `beat_to_beat_bpm[]` | Good (BPM → RR conversion) |
| 3 | `hrv` | Apple SDNN scalar | Limited (time-domain only) |

### 7.2 Ingestion (`POST /v1/ingest`)

1. Split `heartbeat_series` from regular samples; batch-insert the rest.
2. For each `heartbeat_series`: compute `{sdnn, rmssd, pnn50, mean_nn, mean_hr}` via [`compute_hrv_from_rr`](app/hrv_utils.py); persist back into `payload.computed_metrics`.
3. If ≥30 RR samples → calm-score pipeline (§7.4).
4. Backfill `hrv_sdnn` rows lacking `computed_metrics`.

### 7.3 Context Assembly (`fetch_hrv_context_apple` / `fetch_hrv_context`)

- 14-day daily matrix (date → SDNN, mean HR).
- 90-day daily for trend.
- 2-hour buckets (30-day) for circadian patterns.
- 90-day aggregates: mean / p10 / p90 + half-split trend (`improving|stable|declining`, ±5%).
- Top-10 `calm_score_session` summaries.
- Top-10 `mindfulness_sessions` with pre/post HRV deltas.
- Rendered by `_format_hrv_compact` into a compact CSV-like block to minimize tokens.

### 7.4 Calm-Score / Stress State (`hrv_bpm_per_min.process_bpm_session`)

Signal-processing pipeline:
1. Resample irregular BPM → **1 Hz** uniform grid.
2. Clean: physiologic gate 40–180 BPM, spike filter `|Δ|>15`, gap interpolation.
3. Detrend: smoothness-priors (λ=500) with moving-average fallback.
4. Feature extraction in 30-s windows / 5-s cadence:
   - **Baevsky SI proxy** = AMO / (2·MO·MxDMn) — autonomic load index.
   - **HF (0.15–0.40 Hz)** & **LF (0.04–0.15 Hz)** via Welch periodogram → LF/HF.
   - HR trend (60-s mean, slope, deviation).
   - Breath coherence (HF peak ≈ 0.1 Hz / 6 bpm resonance).
5. Per-user adaptive baseline: 60-s init + EMA τ=10 min (`load_cross_session_baseline`).
6. Z-score → weighted calm score 0–100: `0.35*HR + 0.25*SI + 0.30*HF + 0.10*resonance` → `state ∈ {recovery, neutral, stress}`.
7. `SessionSummary`: hr_delta, hf_pct_change, breath_start/end, avg_calm_score, time-in-state %.

NeuroKit2 (`hrv_neurokit.py`) supplies frequency-domain + nonlinear extras (SD1/SD2, sample entropy, DFA α1, RSA) when available; numpy fallback retains the time-domain core.

---

## 8. Mindfulness Sessions (`/v1/mindfulness/session`)

```
iOS posts (beginning_rr[], ending_rr[], duration, mood, depth)
   │
   ├─ filter RR to 200–2000 ms physiological range
   ├─ compute beginning_hrv, ending_hrv, session_hrv (combined)
   ├─ hrv_delta with outcome rule:  improved if ΔSDNN>2 OR ΔRMSSD>3
   ├─ if combined RR ≥30:
   │     process_bpm_session() → snapshots[] + summary
   │     save_session_results() → INSERT health_samples (calm_score_session)
   └─ INSERT mindfulness_sessions
        beginning_hrv / ending_hrv / hrv_delta / session_hrv (JSONB)
        calm_score_ref → health_samples.id
        narration_id → audio_narrations.id (if linked to a generated meditation)
```

---

## 9. Meditation / Audio Generation

### 9.1 Service pipeline (`meditation_service.generate_meditation`)

1. **SSML script** — pick prompt by duration (short / medium / deep), call `chat_once()` to leverage HRV + memories + RAG; output includes `<break time="…"/>` cues.
2. **Voice (parallel)** — primary HuggingFace Gradio space (`HF_SPACE`, voice "Drew"); on failure → ComfyUI `/narration` (returns *pre-merged* MP3, skipping step 5).
3. **Music (parallel)** — ElevenLabs `client.music.compose(prompt, music_length_ms=60000)`; fallback to bundled `app/assets/fallback_music.mp3`.
4. **Title** — `call_gpt_mem0(MEDITATION_TITLE_PROMPT)` → 3–5 word title.
5. **Merge** — pydub/ffmpeg: voice 0 dB, music looped & mixed at −16.5 dB.
6. **Persist** — INSERT `audio_narrations`; enforce per-user cap (25, oldest deleted with file unlink).
7. **Stream** — `GET /v1/practice/audio/stream/{file}` returns `audio/mpeg` `FileResponse`; iOS plays via `AVPlayer`.

### 9.2 Inline trigger from chat
- LLM emits `[GENERATE_MEDITATION]` tag → tag stripped from reply, background task generates audio, then inserts a synthetic message `🧘 Your meditation is ready: [MEDITATION_AUDIO:url:title]` so iOS can render a play button retroactively.

ComfyUI integration spec lives in [Narration_comfyUI_API.md](Narration_comfyUI_API.md).

---

## 10. Calendar Integration

- iOS reads EventKit (past 7 / next 7 days) → `POST /v1/calendar/sync` with `{events[], sync_days, timezone}`.
- Server upserts into `user_calendar_context` (PK `user_uid`, JSONB events).
- During chat, `format_calendar_context` renders a compact block; LLM can suggest schedule-aware actions.
- Reverse signal: `chat_router` keyword-scans replies for calendar verbs ("add to calendar", "scheduled", "moved", "reschedule") → returns `calendar_change=true` so iOS can write back via EventKit.

---

## 11. Authentication

- iOS performs Sign in with Apple → ID token (RS256 JWT) sent in `x-apple-id-token`.
- [`auth.verify_apple_token`](app/auth.py): fetch JWKS (24-h cache by `kid`), verify signature, validate `iss=https://appleid.apple.com`, `aud=APPLE_BUNDLE_ID`, `exp`. `sub` becomes `user_uid`.
- Apple yields name/email **only on first sign-in** → `auth_router.register` upserts `users` immediately.
- Legacy fallback: static `x-app-token` + client-asserted `user_uid` (mobile flows still using this).

---

## 12. Database Schema (PostgreSQL `neuroheart`)

| Table | Key columns |
|---|---|
| `users` | `user_id PK`, email UNIQUE, name, age_range, created_at, last_seen_at |
| `conversations` | `id UUID PK`, user_uid, title, updated_at, is_archived |
| `chat_messages` | `id BIGSERIAL`, conversation_id FK, user_uid, role, content, model, metadata JSONB, created_at |
| `conversation_summaries` | `conversation_id PK`, user_uid, summary, summarized_through_message_id |
| `user_cross_chat_profiles` | `user_uid PK`, profile (multi-line dated), updated_at |
| `health_samples` | id, user_id FK, sample_type, value, unit, source, start_time, end_time, payload JSONB |
| `mindfulness_sessions` | id, user_id, start_time, end_time, duration_minutes, mood, depth, beginning_hrv/ending_hrv/hrv_delta/session_hrv (JSONB), calm_score_ref, calm_summary, narration_id |
| `audio_narrations` | `id UUID PK`, user_uid, conversation_id, session_id, meditation_type, audio_type, file_path, duration_seconds, title, metadata JSONB |
| `user_calendar_context` | `user_uid PK`, events_json JSONB, sync_days, timezone, synced_at |

Migrations live in [migrations/](migrations/). SQL via SQLAlchemy `text()` (no ORM); note CAST syntax: prefer `CAST(:p AS text)` over `:p::text`.

---

## 13. AI / ML Engineering Choices

| Concern | Choice |
|---|---|
| Foundation LLM | OpenAI **gpt-5-nano** (272k input ctx); reasoning model — needs `max_completion_tokens ≥ 16384` |
| Cheap LLM (mem0/summary/title) | `OPENAI_MODEL_MEM0` = gpt-5-nano (cost-budgeted) |
| Embeddings | `text-embedding-3-small` (1536-d), cosine |
| Tokenization | `tiktoken` cl100k_base — slight over-count for newer models; safe for budgeting |
| Retrieval | Qdrant ANN (HNSW), per-user payload filter, top-k cosine |
| Memory dedup | Cosine threshold 0.92 |
| Memory recall | Top-5, score floor 0.20, prefix-dedup |
| Summarization | Threshold-triggered rolling summary, 800-tok cap |
| Prompt engineering | Strict priority ordering, hard 100k budget, graceful degradation of RAG/history |
| Observability | LangSmith via `wrap_openai_client` + `@traceable_call` |
| Voice synth | HuggingFace Gradio space (primary), ComfyUI ElevenLabs (fallback) |
| Music synth | ElevenLabs `music.compose` |
| Signal processing | NeuroKit2 (Welch, DFA, sample entropy, Poincaré); SciPy/NumPy fallbacks |

---

## 14. SWE Principles Applied

- **Layered architecture:** routers (transport) → services (orchestration) → repositories (persistence) → clients (external I/O).
- **Repository pattern:** [`history_repository.py`](app/history_repository.py) centralizes SQL.
- **Singletons for expensive resources:** OpenAI/Qdrant clients; SQLAlchemy engine with `pool_pre_ping`.
- **Dependency injection** via FastAPI `Depends` (`get_verified_user_uid`, `_require_app_token`).
- **Async fan-out + thread offloading** for parallel context assembly.
- **Background tasks** for non-blocking memory write-back and audio generation.
- **Token-bucket rate limiting** (O(1), per-user state).
- **Caching** (Apple JWKS 24h TTL).
- **Graceful degradation:** every external call has a fallback (HF→Comfy, ElevenLabs→bundled, remote HRV→local HRV→empty).
- **Prompt template centralization** in [`app/prompts.py`](app/prompts.py).
- **Structured logging** (`logger.info / warning / exception`).
- **Config as code:** frozen dataclass `Settings` from `.env`.
- **Immutability of secrets:** `APP_TOKEN` constant-time check.

### Data structures & algorithms in evidence
- HNSW ANN (Qdrant) for embedding retrieval.
- Token-bucket fairness for rate-limit.
- Welch periodogram (FFT) for HF/LF spectral power.
- EMA (exponential moving average) for adaptive HRV baseline.
- Smoothness-priors detrending (regularized linear inverse) on BPM signal.
- Half-split trend test (90-d HRV improving/declining/stable).
- Baevsky stress index (histogram-based AMO/MO/MxDMn).
- Sample entropy & DFA α1 (nonlinear complexity).
- Greedy budgeted prompt assembly with priority queue semantics.
- Cosine-similarity dedup (memories, RAG passages).

---

## 15. Frontend / iOS Integration Map

| iOS Subsystem | Server endpoint(s) | Notes |
|---|---|---|
| Sign in with Apple | `/v1/auth/register`, `/v1/auth/me` | `x-apple-id-token` header |
| Chat UI | `/v1/chat/conversations`, `/v1/chat/history`, `/v1/chat` | Pagination via `before_id`; reply may carry `calendar_change` / `generate_meditation` flags |
| HealthKit observer | `/v1/ingest` | Periodic batched upload of `heart_rate`, `hrv`, `hrv_sdnn`, `heartbeat_series`, `sleep`, `steps` |
| Watch meditation | `/v1/mindfulness/session` | Sends pre/post RR arrays |
| Practice / Breathing | `/v1/practice/generate` | Text script |
| Guided meditation | `/v1/practice/generate-meditation`, `/audio/stream/*` | AVPlayer streams MP3 |
| User uploads | `/v1/practice/audio/upload`, `/audio/list`, `DELETE /audio/{id}` | Base64 MP3 in, signed URL out |
| EventKit sync | `/v1/calendar/sync` | Past 7 + next 7 days |

---

## 16. External Integrations

| Service | Purpose | Module |
|---|---|---|
| Apple ID (JWKS) | Verify Sign-in with Apple JWT | [auth.py](app/auth.py) |
| OpenAI (chat + embeddings) | LLM + embeddings | [openai_client.py](app/openai_client.py), [rag_service.py](app/rag_service.py) |
| Qdrant | Vector store (RAG + L2 memory) | [rag_service.py](app/rag_service.py), [memory_service.py](app/memory_service.py) |
| HuggingFace Gradio | Voice TTS (primary) | [meditation_service.py](app/meditation_service.py) |
| ElevenLabs | Music composition + TTS via Comfy | [meditation_service.py](app/meditation_service.py) |
| ComfyUI `/narration` | Pre-merged voice+music fallback | [meditation_service.py](app/meditation_service.py), [Narration_comfyUI_API.md](Narration_comfyUI_API.md) |
| HRV API (port 8002) | Remote HRV analysis (when `HRV_LOCAL=false`) | [hrv_client.py](app/hrv_client.py) |
| LangSmith | LLM tracing | [llm_observability.py](app/llm_observability.py) |
| iOS EventKit | Calendar (via /sync) | [calendar_sync.py](app/calendar_sync.py) |
| iOS HealthKit | HRV / HR / sleep / steps (via /ingest) | [ingest_router.py](app/ingest_router.py) |

---

## 17. Configuration Highlights

| Setting | Default | Purpose |
|---|---|---|
| `MAX_CONTEXT_TOKENS` | 100,000 | Hard prompt budget |
| `CHAT_RECENT_TURNS` | 15 | L1 window |
| `CHAT_SUMMARIZE_THRESHOLD` | 50 msgs | L3 trigger |
| `CHAT_HISTORY_TOKEN_TRIGGER` | 50,000 | L3 trigger (alt) |
| `CHAT_RAG_CHUNK_TOKENS` / `MAX_CHUNKS` | 300 / 20 | RAG budget |
| `MEMORY_RETRIEVAL_TOP_K` | 5 | L2 recall |
| `MEMORY_DUPLICATE_THRESHOLD` | 0.92 | L2 dedup |
| `MEMORY_MAX_PER_USER` | 200 | L2 cap |
| `MEM0_MAX_COST` | $2/day | L2 daily budget |
| `RATE_LIMIT_CAPACITY / REFILL` | 20 / 0.333 tps | Per-user limit |
| `MEDITATION_MAX_STORED` | 25 | Per-user audio cap |
| `HRV_LOCAL` | true | Use Apple data over remote API |

---

## 18. Deployment

```bash
# VPS (159.198.44.98)
cd /opt/neuroheart/chat-api
git pull origin main
source .venv/bin/activate
pip install -r requirements.txt
systemctl restart neuroheart-chat
journalctl -u neuroheart-chat -f
```

Notes:
- VPS Python 3.12 requires `numpy<2.0` (no AVX2 on host CPU).
- Postgres local auth: `sudo -u postgres psql -d neuroheart` (password auth for `neuroheart_user` is broken on VPS).
- Audio storage at `/opt/neuroheart/audio` served behind Nginx as `https://neuroheart.ai/audio/...`.

---

## 19. Cross-Cutting Concerns

- **Security:** Apple JWT verification with rotating JWKS, constant-time `APP_TOKEN` compare, file-name sanitation in audio stream, per-user payload filters in Qdrant, prepared statements via SQLAlchemy `text()`.
- **Cost control:** cheap-model split (`OPENAI_MODEL_MEM0`), daily mem0 budget, RAG/history graceful shrink, summarization to keep window bounded.
- **Resilience:** every external dependency has a no-context fallback path; LLM errors raise but do not corrupt DB state (user message persisted before LLM call; assistant message only on success).
- **Observability:** LangSmith tracing on `chat_completion`; structured logs include token breakdown, latency, RAG hit count.
- **Privacy:** raw RR intervals stored only in `health_samples.payload`; aggregates in summaries; user can delete meditations and (by extension) cascade audio files.

---

## 20. Glossary of NeuroHeart-Specific Terms

- **Calm score** — 0–100 scalar from `process_bpm_session` blending HR, Baevsky SI, HF power, and 0.1 Hz resonance against a per-user adaptive baseline.
- **Cross-chat profile** — dated 1-line digest per conversation, retained 30 days, separate from L2 memories.
- **Layer-3 summary** — per-conversation rolling summary triggered by message count or older-window token mass.
- **Tier 1/2/3 HRV** — data-quality stratification of HealthKit samples (raw RR > BPM list > Apple SDNN).
- **`[GENERATE_MEDITATION]`** — sentinel emitted by the LLM to trigger background audio generation, stripped before display.
