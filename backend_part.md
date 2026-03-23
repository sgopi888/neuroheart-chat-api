# Practice + Chat Meditation Integration Plan

## Context
Three phases of work: (1) Fix playlist font sizes + unique titles, (2) Add meditation generation from Chat tab, (3) Long-press TTS on chat bubbles.

---

## Phase 1: Playlist Fixes

### 1A. Double font sizes in past sessions list
**File:** `NeuroHeartNew.swift` — `pastPracticesSection`
- Title: 15pt → 18pt (displaySerifItalic)
- Duration/date/time: 13pt → 15pt (dmSans)
- Play icon: 14pt in 42pt circle → 16pt in 48pt circle
- Trash icon: 14pt → 16pt

### 1B. Unique titles per session
**Problem:** All titles show same "Stressed · Breathing" because `displayTitle` uses metadata mood+sessionType which are often identical.

**Solution — Backend:** In `meditation_service.py`, after generating the SSML script, ask the LLM for a short unique title (3-5 words) based on the script content. Store in `title` field of `audio_narrations`.

**Solution — Frontend fallback:** Update `AudioNarrationItem.displayTitle` to include date/time if mood+type are the same pattern:
```swift
var displayTitle: String {
    if let mood, let sessionType {
        return "\(mood) · \(sessionType.capitalized)"
    }
    if let t = title, !t.isEmpty { return t }
    return "Meditation"
}
```
This is already fine — the real fix is backend generating unique titles. For now, frontend can append a short time suffix: `"Stressed · Breathing · 3:45 PM"`

### 1C. Include chat history in meditation generation
**Already done.** Backend `chat_once()` loads last 10 conversation turns + rolling summary + long-term memories + HRV + RAG when generating the meditation script. The conversation_id passed from frontend links to the correct history.

### 1D. Post meditation link as AI message in chat
After meditation is generated, send the `audio_url` as an assistant message into the conversation so it appears in chat history.

**Backend change needed:** In `meditation_router.py` after generating, insert an assistant message into the conversation with the audio link + title.

**Frontend:** Detect messages containing audio URLs and render them as playable meditation cards (not plain text).

---

## Phase 2: Generate Meditation from Chat Tab

### 2A. Generate button in chat header
**File:** `ChatThreadView.swift` — `.toolbar` section

Add a meditation icon button next to the header, in `.navigationBarTrailing`:
```swift
ToolbarItem(placement: .navigationBarTrailing) {
    Button { showMeditationSheet = true } label: {
        Image(systemName: "waveform.circle")
            .foregroundStyle(Color.accentPrimary)
    }
}
```

Tapping shows a small sheet/menu with mood + duration picker (reuse the Quick mode chips from PracticeTabView), then generates using the CURRENT chat's `conversationID`. This ensures the meditation is personalized to that chat's history.

### 2B. Auto-trigger via LLM response flag
**Backend changes needed:**
1. Add `generate_meditation: bool` field to `ChatResponse` in `schemas.py`
2. In `chat_router.py`, detect when LLM suggests meditation (keyword detection or instruction in system prompt to output a JSON flag)
3. When `generate_meditation: true`, backend auto-generates meditation and returns `audio_url` + `session_id` in the response

**Frontend changes:**
1. Add `generate_meditation: Bool?` to `ChatResponse` in `ChatModels.swift`
2. In `ChatViewModel.sendMessage()`, after receiving response: if `generate_meditation == true`, trigger meditation generation using `ScriptGenerationService` with the current `conversationID`

### 2C. User text trigger ("generate a meditation for me")
Same as 2B — the backend LLM detects the user intent and sets `generate_meditation: true` in the response. No separate frontend logic needed.

### 2D. Meditation appears as chat message + opens in Practice tab
After generation:
1. An AI message with the meditation link appears in the chat thread
2. The message is rendered as a tappable card (title, duration, play icon)
3. Tapping the card switches to Practice tab and plays the audio using `MeditationAudioPlayer`
4. The same meditation also appears in Practice tab's past sessions list (it's in `audio_narrations` table)
5. If user returns to Chat tab, the link message is still visible in history

**Message format:** The AI message content could be:
```
[MEDITATION:session_id:audio_url:title]
Here's your personalized meditation: "Calm Evening Breathwork" (5 min)
```
Frontend parses this to render a playable card.

---

## Phase 3: Long-press TTS on Chat Bubbles

### 3A. Add "Play" option to context menu
**File:** `MessageBubble.swift`

The context menu already has "Copy" and "Speak" options. "Speak" already calls `speechManager.speak(message.content)` which uses iOS built-in AVSpeechSynthesizer.

**Current state:** This already works! The `onSpeak` callback in MessageBubble calls `ChatViewModel.speakMessage()` which uses `SpeechManager.speak()`.

**Verify:** The long-press context menu shows "Speak" for assistant messages. If it's only visible as a button below the bubble, add it to the `.contextMenu` as well for discoverability.

---

## Files to Modify

### Frontend (iOS)
| File | Changes |
|------|---------|
| `NeuroHeartNew.swift` | Font size increase in pastPracticesSection, title with time suffix |
| `ChatModels.swift` | Add `generate_meditation: Bool?` to ChatResponse, meditation message model |
| `ChatThreadView.swift` | Add meditation button in toolbar, detect meditation messages |
| `ChatViewModel.swift` | Handle `generate_meditation` flag, trigger generation |
| `MessageBubble.swift` | Render meditation link messages as playable cards, ensure Speak in context menu |
| `AppState.swift` | May need method to switch to Practice tab + play audio |

### Backend (Python) — tell backend dev
| File | Changes |
|------|---------|
| `app/schemas.py` | Add `generate_meditation: bool = False` to ChatResponse |
| `app/chat_router.py` | Detect meditation trigger, auto-generate, insert message |
| `app/meditation_service.py` | Generate unique titles via LLM |
| `app/prompts.py` | Add instruction to system prompt about meditation triggering |

---

## Implementation Order
1. **Phase 1** first (font fixes, title improvements) — frontend only, quick
2. **Phase 3** next (long-press TTS) — verify existing, small tweak
3. **Phase 2** last (chat integration) — needs backend changes, most complex

---

## Backend Developer Instructions (hand off separately)

### 1. Unique meditation titles
**File:** `app/meditation_service.py`
After generating the SSML script, call LLM for a short contextual title (3-5 words) instead of hardcoding `f"Meditation: {mood} {date}"`. Use the script excerpt + mood as input. Store the unique title in `audio_narrations.title`.

### 2. Add `generate_meditation` flag to chat response
**File:** `app/schemas.py`
```python
class ChatResponse(BaseModel):
    # ... existing fields ...
    generate_meditation: bool = False
```

**File:** `app/chat_router.py`
After calling `chat_once()`, detect if the user asked for meditation OR if the LLM suggests one. Options:
- Keyword detection in user message: "generate meditation", "can you make me a meditation", "breathing exercise", etc.
- OR add instruction to system prompt telling LLM to include `[GENERATE_MEDITATION]` tag in its reply when appropriate, then parse it out server-side

When detected, set `generate_meditation = True` in response.

### 3. Auto-generate meditation when triggered
**File:** `app/chat_router.py`
When `generate_meditation` is true:
1. Call `generate_meditation()` from `meditation_service.py` using the same `conversation_id` (so it gets the chat history context)
2. Extract `audio_url` and `title` from the result
3. Insert an assistant message into the conversation: `"Here's your meditation: {title}\n[MEDITATION_AUDIO:{audio_url}]"`
4. Return the normal chat reply + set `generate_meditation: true` so frontend knows

### 4. Post meditation link as assistant message
**File:** `app/meditation_router.py` or `app/chat_router.py`
After `generate_meditation()` returns, insert a new row in `chat_messages` with:
- `conversation_id`: same as current chat
- `role`: "assistant"
- `content`: formatted string with audio URL that frontend can parse
- This makes the meditation link appear in chat history even if user reloads

### 5. System prompt update for meditation awareness
**File:** `app/prompts.py`
Add to `CHAT_SYSTEM_PROMPT`:
```
When the user asks you to generate, create, or make a meditation, breathing exercise,
or mindfulness practice, include the tag [GENERATE_MEDITATION] at the end of your reply.
Do not include this tag unless the user specifically asks for a meditation to be generated.
```

### Summary for backend dev
| Task | File | Priority |
|------|------|----------|
| Unique LLM-generated titles per meditation | `meditation_service.py` | HIGH |
| Add `generate_meditation: bool` to ChatResponse | `schemas.py` | HIGH |
| Detect meditation request in chat + set flag | `chat_router.py` | HIGH |
| Auto-generate meditation when flag is set | `chat_router.py` | HIGH |
| Insert meditation link as assistant message | `chat_router.py` | HIGH |
| Update system prompt with meditation tag instruction | `prompts.py` | MEDIUM |

---

## Verification
1. Past sessions: fonts visibly larger, each title unique, date+time shown
2. Chat: meditation button visible in header → generates → link appears in chat → tappable → opens in Practice tab
3. Chat: type "generate a meditation" → LLM responds → meditation auto-generated → link in chat
4. Chat: long-press any assistant message → "Speak" option → iOS TTS reads it
5. Practice tab: all generated meditations (from chat + practice) appear in past sessions
