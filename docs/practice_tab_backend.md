# Practice Tab Backend Changes (neuroheart-chat-api)

## Context
The frontend Practice tab will send structured parameters for meditation script generation. The backend needs a dedicated endpoint (or enhanced chat endpoint) and a centralized prompts file to handle practice-specific generation.

---

Remember, the chat functions, hsitory, cal, all are worokign well, this is to support practice tab and also some modulairation of prompts.

## 1. New File: `app/prompts.py` — Centralized Prompts

All prompt templates in one file for easy editing.

```python
"""Centralized prompt templates for all LLM interactions."""

# ── Practice / Meditation ──────────────────────────────────────────

PRACTICE_QUICK_PROMPT = """
Based on the user's recent HRV data, chat history, and memory, generate a guided {session_type} meditation script.
The user is currently feeling: {mood}.
Duration: {duration} minutes.
Output plain text, one instruction per line. Include breathing cues, pauses (as "..."), and gentle transitions.
Keep the tone warm, calm, and supportive. Tailor to their emotional state.
"""

PRACTICE_DEEP_PROMPT = """
Based on the user's recent HRV data, chat history, and memory, generate a guided {session_type} meditation script.
The user is currently feeling: {mood}.
Focus area: {depth}.
Duration: {duration} minutes.
Output plain text, one instruction per line. Include breathing cues, pauses (as "..."), and gentle transitions.
Explore the {depth} focus deeply — guide the user through introspection on this theme.
Keep the tone warm, calm, and supportive. Tailor to their emotional state and focus area.
"""

DAILY_RECOMMENDATION_PROMPT = """
Provide a single-sentence daily practice recommendation based on the user's recent health data and patterns.
"""

# ── Existing Chat System Prompt (move here from chat_service.py) ──

CHAT_SYSTEM_PROMPT = """
You are NeuroHeart, a personal health insights assistant.
When the user asks to create, modify, move, or cancel a calendar event,
respond with a friendly confirmation that includes the event title and time.
...
"""

# ── Memory Extraction (move here from chat_service.py) ── and connect so that it works same; 

MEMORY_EXTRACTION_PROMPT = """..."""

# ── Summarization (move here from chat_service.py) ── and connect so that it works same; 

SUMMARIZATION_PROMPT = """..."""
```

**Action:** Move all hardcoded prompt strings from `chat_service.py` (lines 48-61 system prompt, summarization prompt, memory extraction prompt) into this file.

---

## 2. New Endpoint: `POST /v1/practice/generate`

Add a dedicated practice generation endpoint (or handle via existing chat with metadata).

### Option A: Dedicated Endpoint (Recommended)

**File:** `app/practice_router.py` (new)

```python
from fastapi import APIRouter
router = APIRouter(prefix="/v1/practice", tags=["practice"])

class PracticeRequest(BaseModel):
    user_uid: str
    conversation_id: str
    mood: str                          # "Stressed", "Sad", "Anxious", "Depressed"
    depth: str | None = None           # "Mind", "Body", "Emotions", "Spirit", "Past", "Future"
    duration: int                      # minutes: 2, 5, 10, 20, 40
    session_type: str = "breathing"    # "breathing", "meditation", "body scan"

class PracticeResponse(BaseModel):
    conversation_id: str
    script: str                        # generated meditation script
    title: str                         # "Practice: Stressed 2026-03-20"

@router.post("/generate", response_model=PracticeResponse)
async def generate_practice(req: PracticeRequest):
    # 1. Select prompt template based on whether depth is provided
    # 2. Format prompt with mood, depth, duration, session_type
    # 3. Call chat_service.chat_once() with the formatted prompt
    #    (this gives access to HRV context, memories, RAG — all the good stuff)
    # 4. Store user message + assistant reply in the conversation
    # 5. Return script
```

**Register in `main.py`:**
```python
from app.practice_router import router as practice_router
app.include_router(practice_router)
```

### Option B: Use Existing Chat Endpoint with Metadata
The frontend constructs the message like: `"[PRACTICE] mood=Stressed depth=Mind duration=10 type=breathing"` and the backend detects the `[PRACTICE]` prefix to use the practice prompt template. Less clean but no new endpoint needed.

**Recommendation:** Option A — cleaner separation, easier to evolve independently.

---

## 3. Update `app/chat_service.py`

### Import prompts from `prompts.py`
```python
from app.prompts import CHAT_SYSTEM_PROMPT, PRACTICE_QUICK_PROMPT, PRACTICE_DEEP_PROMPT
```

### Add practice generation function
```python
async def generate_practice_script(
    user_uid: str,
    conversation_id: str,
    mood: str,
    depth: str | None,
    duration: int,
    session_type: str,
) -> str:
    """Generate meditation script using full context (HRV, memories, RAG)."""
    # Select and format prompt
    if depth:
        prompt = PRACTICE_DEEP_PROMPT.format(
            mood=mood, depth=depth, duration=duration, session_type=session_type
        )
    else:
        prompt = PRACTICE_QUICK_PROMPT.format(
            mood=mood, duration=duration, session_type=session_type
        )

    # Reuse existing chat_once() flow for full context (HRV, memories, RAG)
    # but with the practice-specific prompt as the user message
    response = await chat_once(
        user_uid=user_uid,
        conversation_id=conversation_id,
        user_message=prompt,
        hrv_range="7d",
    )
    return response["reply"]
```

### Move hardcoded prompts to `prompts.py`
- System prompt (lines 48-61) → `CHAT_SYSTEM_PROMPT`
- Summarization prompt → `SUMMARIZATION_PROMPT`
- Memory extraction prompt → `MEMORY_EXTRACTION_PROMPT`
- Cross-chat profile prompt → `CROSS_CHAT_PROFILE_PROMPT`

---

## 4. Conversation Title Support

The `POST /v1/chat/conversations` endpoint already accepts `title` in the request body (`CreateConversationRequest.title`). The frontend will send titles like "Practice: Stressed 2026-03-20". No backend changes needed for this.

Verify in `app/history_repository.py` that `create_conversation()` stores the title in the `conversations` table.

---

## 5. List Conversations Filtering (Optional)

Consider adding a `prefix` query param to `GET /v1/chat/conversations`:

```python
@router.get("/v1/chat/conversations")
async def list_conversations(user_uid: str, prefix: str | None = None):
    # If prefix provided, filter WHERE title LIKE '{prefix}%'
    # This avoids loading ALL conversations just to filter client-side
```

Not critical for v1 — frontend can filter client-side.

---

## Summary of Backend Changes

| Change | File | Priority |
|--------|------|----------|
| Create `prompts.py` with all prompt templates | `app/prompts.py` (new) | HIGH |
| Create practice generation endpoint | `app/practice_router.py` (new) | HIGH |
| Add practice generation function | `app/chat_service.py` | HIGH |
| Register practice router | `app/main.py` | HIGH |
| Move hardcoded prompts to `prompts.py` | `app/chat_service.py` | MEDIUM |
| Optional: prefix filter for conversations | `app/chat_router.py` | LOW |

---

## Frontend Integration

Once the backend `POST /v1/practice/generate` endpoint is ready, the iOS `ScriptGenerationService.swift` will:
1. Create conversation via `POST /v1/chat/conversations` with title "Practice: {mood} {date}"
2. Call `POST /v1/practice/generate` with conversation_id, mood, depth, duration, session_type
3. Receive script in response
4. Pass script to PracticeTTSManager for playback

**Until the backend endpoint is ready**, the frontend can use the existing `POST /v1/chat` endpoint with a client-side formatted prompt (current behavior, just improved with mood/depth params).
