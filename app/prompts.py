"""Centralized prompt templates for all LLM interactions."""

# ── Practice / Meditation ──────────────────────────────────────────

PRACTICE_QUICK_PROMPT = """
Based on the user's recent HRV data, chat history, and memory, generate a guided {session_type} meditation script.
The user is currently feeling: {mood}.
Duration: {duration} minutes.
Output plain text, one instruction per line. Include breathing cues, pauses (as "..."), and gentle transitions.
Keep the tone warm, calm, and supportive. Tailor to their emotional state.
""".strip()

PRACTICE_DEEP_PROMPT = """
Based on the user's recent HRV data, chat history, and memory, generate a guided {session_type} meditation script.
The user is currently feeling: {mood}.
Focus area: {depth}.
Duration: {duration} minutes.
Output plain text, one instruction per line. Include breathing cues, pauses (as "..."), and gentle transitions.
Explore the {depth} focus deeply — guide the user through introspection on this theme.
Keep the tone warm, calm, and supportive. Tailor to their emotional state and focus area.
""".strip()

DAILY_RECOMMENDATION_PROMPT = """
Provide a single-sentence daily practice recommendation based on the user's recent health data and patterns.
""".strip()

# ── Chat System Prompt ─────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = (
    "You are NeuroHeart, a personal health insights assistant. "
    "When the user asks to create, modify, move, or cancel a calendar event, "
    "respond with a friendly confirmation that includes the event title and time. "
    "For example: \"Sure, I'll add 'Meditation' to your calendar for tomorrow at 9:00 AM.\" "
    "or \"I'll cancel your workout on Friday.\" "
    "Always use phrases like \"I'll add to your calendar\" or \"I've scheduled\" "
    "so the system can detect the action. "
    "Use the provided HRV and health context when relevant to give personalized advice. "
    "Use the context but be kind and like a counsellor and expert mindfulness expert. "
    "Never reference another user's data. "
    "Keep answers concise, practical, and supportive."
)

# ── Summarization ──────────────────────────────────────────────────

SUMMARIZATION_SYSTEM = "You are a precise summarizer."

SUMMARIZATION_USER_TEMPLATE = (
    "You are summarizing a health coaching chat session.\n\n"
    "{existing_summary_block}"
    "NEW MESSAGES TO INCORPORATE:\n{history_text}\n\n"
    "Produce a concise updated summary in this structured format:\n"
    "- User profile / preferences learned:\n"
    "- Recurring symptoms / patterns mentioned:\n"
    "- Key events / dates:\n"
    "- Action items / commitments:\n"
    "- Open questions / unresolved threads:\n"
    "Keep each section to 1-3 bullet points maximum."
)

# ── Memory Extraction ──────────────────────────────────────────────

MEMORY_EXTRACTION_PROMPT = """\
Extract key facts from this chat exchange that would be useful to remember \
for future conversations with this user. Focus on:
- Health conditions, symptoms, diagnoses mentioned
- User preferences (exercise habits, sleep schedule, diet, meditation practice)
- Personal context (job, stress triggers, goals, family situation)
- Decisions made or commitments (e.g. "will try breathing exercises before bed")
- Patterns noticed (e.g. "HRV drops on work-from-home days")

Output ONLY a JSON array of strings, each a concise standalone fact.
If nothing worth remembering, output [].
Example: ["User practices meditation 3x/week", "Has insomnia on Sunday nights"]

CHAT EXCHANGE:
User: {user_msg}
Assistant: {assistant_msg}
"""

# ── Cross-chat Profile ─────────────────────────────────────────────

CROSS_CHAT_PROFILE_PROMPT = """\
Add ONE short line (10-20 words max) summarizing what was discussed in this chat session.
This line will be appended to the user's memory profile.

Format: "YYYY-MM-DD: <one-liner about what was discussed>"

CHAT EXCHANGE:
User: {user_msg}
Assistant: {assistant_msg}

Output ONLY the single dated line, nothing else.
Example: "2026-03-14: Discussed 3am wake-ups linked to work stress, suggested body scan before bed"
"""
