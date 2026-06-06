# NeuroHeart AI — Apple Approval Remediation: Backend Change Log

Tracks every code edition made in the `neuroheart-chat-api` repo for the Apple
approval remediation (repositioning NeuroHeart as a wellness / mindfulness /
lifestyle app, not a medical app).

Scope of this pass (per direction): **Backend + safety first.** Database row
purging is executed on the VPS after SSH; only application code changes live in
this repo. Copy is aligned to the **NeuroHeart AI Privacy Policy (June 2026)**.

---

## Phase 1 — AI Safety & Scope Control

### 1.1 / 1.2 — System prompt rewritten for wellness scope
**File:** `app/prompts.py` → `CHAT_SYSTEM_PROMPT` (+ new `OUT_OF_SCOPE_RESPONSE`)

- Repositioned the assistant as a "wellness and mindfulness companion," explicitly
  **not** a medical provider.
- Added an **ALLOWED TOPICS** list (stress management, mindfulness, meditation,
  breathing, sleep habits, relaxation, gratitude, journaling, wellness coaching,
  habit formation, recovery/self-care).
- Added an **OUT-OF-SCOPE TOPICS** list (diagnosis, symptom interpretation, fever
  evaluation, disease identification, medication recommendations/dosing, treatment
  plans, clinical/emergency advice) with instruction to redirect rather than answer.
- Added the canonical out-of-scope reply as a reusable constant
  `OUT_OF_SCOPE_RESPONSE` (also embedded verbatim in the prompt).
- Explicitly forbade medical actions previously possible (check temperature, take
  medication, monitor fever, drink electrolytes for illness, evaluate symptoms);
  the model now offers a wellness alternative instead.
- **Preserved** existing functional hooks: calendar-action confirmation phrases and
  the `[GENERATE_MEDITATION]` tag (the app depends on these for detection).

### 1.3 — Emergency / crisis detection (deterministic guardrail)
**New file:** `app/safety.py`
**File:** `app/prompts.py` → new constant `EMERGENCY_RESPONSE`
**File:** `app/chat_router.py` → guardrail wired into `POST /v1/chat`

- `detect_emergency(text)` — high-precision, word-boundary regexes for suicide,
  self-harm, overdose, stroke, heart attack, seizure, can't-breathe, chest pain,
  unconscious, severe bleeding, 911, etc. Conservative by design to avoid false
  positives on ordinary wellness language (verified: "kill this bad habit",
  "end my stressful day", "heart rate was high during my run" do NOT trigger).
- `looks_out_of_scope(text)` — softer hint for non-emergency medical requests
  (symptom/medication/diagnosis). Advisory; the system prompt handles the redirect.
- In `POST /v1/chat`, **before** the LLM call: if `detect_emergency` matches, the
  endpoint returns `EMERGENCY_RESPONSE` directly (crisis-line guidance: 911 / 988),
  persists both the user turn and the safety reply
  (`metadata.type = "safety_emergency"`), and generates **no** coaching content.

---

## Phase 5 — Account Deletion (backend code)

**New file:** `app/account_deletion.py`
**File:** `app/auth_router.py` → new endpoint `DELETE /v1/auth/me` (+ `_verify_identity`
helper refactor shared by GET/PUT/DELETE `/me`)
**New migration:** `migrations/007_account_deletion.sql`

### Confirmed schema (verified via `psql` on VPS)

| Table | User key | Cascades from `users`? |
|---|---|---|
| `users` | `user_id` | (root) |
| `health_samples` | `user_id` | ✅ CASCADE |
| `mindfulness_sessions` | `user_id` | ✅ CASCADE |
| `audio_narrations` | `user_uid` | ❌ delete explicitly |
| `chat_messages` | `user_uid` | ❌ delete explicitly |
| `conversations` | `user_uid` | ❌ delete explicitly |
| `conversation_summaries` | `user_uid` | ❌ delete explicitly |
| `user_calendar_context` | `user_uid` | ❌ delete explicitly |
| `user_cross_chat_profiles` | `user_uid` | ❌ delete explicitly |

### What the endpoint does (single transaction, idempotent)
1. Writes a minimal security-retention row to `deleted_accounts`
   (`user_id`, `deleted_at`, `purge_after`, `reason`) — **no content** (Task 5.5).
2. Deletes the six `user_uid` content tables explicitly.
3. Deletes the `users` row → cascades `health_samples` + `mindfulness_sessions`.

Auth: Apple ID token (preferred) or legacy app-token + `user_uid` fallback —
same pattern as the existing `/me` endpoints.

### 180-day retention table (Task 5.5, aligned to Privacy Policy §8)
`migrations/007_account_deletion.sql` creates `deleted_accounts` with a 180-day
`purge_after` default (policy says limited security/audit records retained up to
180 days). **No FK to `users`** so it survives the user-row deletion.
➡️ **Run this migration on the VPS before deploying the endpoint.**

### Out of scope for this repo (documented TODOs in `account_deletion.py`)
- **Apple Sign-In token revocation** — this backend authenticates via the Apple
  ID token directly and stores no Apple refresh token, so nothing to revoke
  server-side. (No Firebase Auth in this backend.)
- **Vector memories (Qdrant)** — if long-term memories are persisted to a Qdrant
  collection, add a per-user point purge. Verify collection naming on VPS.
- **On-disk audio files** under `AUDIO_STORAGE_DIR` referenced by
  `audio_narrations` — DB rows removed; orphaned files need a separate sweep.

---

## Verification done in this pass
- `python -m py_compile` passes for all changed files.
- Safety guardrail unit-checked: 8/8 emergency phrases trigger, 6/6 benign
  wellness phrases do not, 4/4 medical out-of-scope phrases detected.

## To run on the VPS (not done here)
1. Apply `migrations/007_account_deletion.sql`.
2. Deploy the updated app and restart the service.
3. (Optional) schedule a sweep to delete `deleted_accounts` rows past `purge_after`
   and to clean orphaned audio files.

---

## Provider note (for later iOS consent / privacy copy — Phases 2/3/6)

The original plan lists only **OpenAI** + **ElevenLabs**. Code in this repo shows
the user-facing third-party processors are actually:

- **OpenAI** — chat replies, summarization, memory extraction, embeddings.
- **ElevenLabs** — voice / TTS narration.
- **Hugging Face Space** (`NeuroHeart2026/voice-agent`, via `gradio_client`) —
  voice narration (alternate path).
- **LangSmith** — LLM observability/tracing (prompts/replies may be logged).
- Qdrant — vector DB for RAG (infrastructure; stores embeddings).

The published Privacy Policy already phrases this **generically** ("third-party
artificial intelligence service providers"), which matches the guidance to NOT
hard-code a fixed provider list. Keep iOS consent/settings copy generic too
(e.g. "AI service providers such as OpenAI, ElevenLabs, and similar").
