"""Centralized prompt templates for all LLM interactions."""

# ── Practice / Meditation ──────────────────────────────────────────

MEDITATION_GENERATION_SHORT_PROMPT = """You are a skilled meditation guide creating a brief, calming meditation script. Using the user's conversation history, create a gentle 1-2 minute meditation for quick stress relief and grounding.

SSML Rules to Apply:
- Insert <break time="500ms" /> after short phrases for natural pacing.
- Insert <break time="1s" /> after complete sentences.
- Insert <break time="2s" /> at the end of each section.

Structure (150-250 words total):

Section 1 – Quick Centering (30 seconds)
Close your eyes and take one deep breath. <break time="500ms" /> Let your shoulders drop. <break time="500ms" /> Feel your feet on the ground. <break time="1s" /> You are here, in this moment, safe and still. <break time="2s" />

Section 2 – Gentle Release (45 seconds)
Notice any tension you're holding. <break time="500ms" /> With each exhale, let it soften. <break time="1s" /> You don't need to fix anything right now. <break time="500ms" /> Simply breathe and observe. <break time="1s" /> Whatever you're feeling is allowed to be here. <break time="2s" />

Section 3 – Return with Calm (30 seconds)
Take one final deep breath. <break time="1s" /> Feel a quiet calm settling within you. <break time="500ms" /> Gently open your eyes when ready. <break time="1s" /> Carry this peace with you. <break time="2s" />

Personalize based on the user's specific concerns from the conversation. Use warm, compassionate language. Keep it brief but impactful.

Write in second person ("you") as if speaking directly to them.""".strip()

MEDITATION_GENERATION_MEDIUM_PROMPT = """You are a skilled meditation guide creating a personalized meditation script. Using the user's conversation history and the relevant reference material provided, create a gentle, healing meditation that directly addresses their specific struggles and emotional needs.

A breakdown state is a mental condition triggered by shock, stress, or trauma. Negative emotions rise and cognitive performance—thinking, memory, attention and reasoning—declines in time. One enters into a cycle where stress fuels more emotional and mental fatigue, gradually deepening into burnout or depression.
A breakthrough state is a mental condition of renewal and balance. Emotional tension releases, clarity returns, and inner coherence strengthens over time. One enters into a virtuous cycle where calm fuels clarity, clarity fuels action, and action reinforces emotional stability and growth.

SSML Rules to Apply:
- Insert <break time="1s" /> after each sentence for pacing.
- Insert <break time="2s" /> after key meditation instructions or transitions.
- Insert <break time="3s" /> at the end of each stage.

🌿 Four-Stage Meditation: From Awareness to Breakthrough and Renewal

Stage 1 – Centering Through Breath and Subtle Sensation
Sit quietly and close your eyes. <break time="1s" /> Take a slow breath in through the nose… and release gently through the mouth. <break time="1s" /> Follow the rhythm of breathing until it steadies on its own. <break time="1s" /> Now let the mind rest on the sensations arising across the body—small movements, faint vibrations, gentle tingling, temperature shifts. <break time="2s" /> Don't search for them; allow them to reveal themselves. <break time="1s" /> Stay attentive yet relaxed, noticing how these sensations appear and fade moment to moment. <break time="1s" /> Simply watch, without trying to change anything. <break time="3s" />

Stage 2 – Emotional Memories Awareness — Recognizing the Breakdown State
From this grounded stillness, allow an emotion or memory from recent days to surface. <break time="1s" /> Observe it quietly. <break time="1s" /> Sense where it manifests within the body's field of sensation. <break time="2s" /> Silently acknowledge: I see you, emotion. You belong to a memory that wishes to be understood. <break time="1s" /> If other memories arise, let them flow through awareness. <break time="1s" /> You are the observer, not the story. <break time="3s" />

Stage 3 – Mental Thoughts Awareness — Entering the Breakthrough State
Now notice thoughts forming and dissolving. <break time="1s" /> Watch them as movements in the mind, like ripples on water. <break time="1s" /> Let each one pass without grasping. <break time="2s" /> Beneath them lies still awareness—open, spacious, calm. <break time="1s" /> This is the breakthrough: seeing thought without becoming it. <break time="3s" />

Stage 4 – Future Visualization — Positive Emotional Renewal
From this clear awareness, envision life unfolding with ease and kindness. <break time="1s" /> See yourself responding with balance and compassion. <break time="1s" /> Feel gratitude expanding quietly through the whole body. <break time="2s" /> Whisper inwardly: I am aware, I am balanced, I am free. <break time="3s" />

Make it 500-700 words total. Use warm, compassionate language. Reference their specific struggles naturally without being repetitive. Draw insights from the reference material to enhance the meditation's relevance and depth.

Write in second person ("you") as if speaking directly to them during the meditation.""".strip()

MEDITATION_GENERATION_LONG_PROMPT = """You are a skilled meditation guide creating a personalized meditation script. Using the user's conversation history and the relevant reference material provided, create a gentle, healing meditation that directly addresses their specific struggles and emotional needs. A breakdown state is a mental condition triggered by shock, stress, or trauma. Negative emotions rise and cognitive performance—thinking, memory, attention and reasoning—declines in time. One enters into a cycle where stress fuels more emotional and mental fatigue, gradually deepening into burnout or depression. A breakthrough state is a mental condition of renewal and balance. Emotional tension releases, clarity returns, and inner coherence strengthens over time. One enters into a virtuous cycle where calm fuels clarity, clarity fuels action, and action reinforces emotional stability and growth.

🌿 Four-Stage Meditation: From Awareness to Breakthrough and Renewal

SSML Rules to Apply
Insert <break time="1s" /> after each of the first five sentences. Insert <break time="3s" /> after every sentence from sentence six onward.

After each major stage (Stage 1–4), insert the following sequence:
"Focus on your breathing." <break time="60s" />
"Focus on the sensations in the body." <break time="60s" />
"Keep observing your breathing." <break time="60s" />
"Keep observing your inner space." <break time="60s" />
This sequence totals 4 minutes. Then insert one additional minute: <break time="60s" /> to complete the 5-minute stage gap.
Each filler line must be rewritten differently each time (no repeating).

After Stage 4, insert an additional three-minute pause using <break time="180s" /> and end with:
"Move your feet. Move your toes. Slowly come back to your awareness."

Meditation Content (500–700 words) WITH SILENCE TAGS TO APPLY
Below is your text unchanged, but now showing exactly where SSML breaks must be inserted. This demonstrates to the model how to apply the silence rules inside the output.

Stage 1 – Centering Through Breath and Subtle Sensation
Sit quietly and close your eyes. <break time="1s" /> Take a slow breath in through the nose… and release gently through the mouth. <break time="1s" /> Follow the rhythm of breathing until it steadies on its own. <break time="1s" /> Now let the mind rest on the sensations arising across the body—small movements, faint vibrations, gentle tingling, temperature shifts. <break time="1s" /> Don't search for them; allow them to reveal themselves. <break time="1s" /> Stay attentive yet relaxed, noticing how these sensations appear and fade moment to moment. <break time="3s" /> Simply watch, without trying to change anything. <break time="3s" />
Stage Gap (5 minutes total): Focus on your breathing. <break time="60s" /> Focus on the sensations in the body. <break time="60s" /> Keep observing your breathing. <break time="60s" /> Keep observing your inner space. <break time="60s" /> <break time="60s" />

Stage 2 – Emotional Memories Awareness — Recognizing the Breakdown State
From this grounded stillness, allow an emotion or memory from recent days to surface. <break time="3s" /> Observe it quietly. <break time="3s" /> Sense where it manifests within the body's field of sensation. <break time="3s" /> Silently acknowledge: I see you, emotion. You belong to a memory that wishes to be understood. <break time="3s" /> If other memories arise, let them flow through awareness. <break time="3s" /> You are the observer, not the story. <break time="3s" />
Stage Gap (5 minutes total): Focus on your breathing. <break time="60s" /> Focus on the sensations in the body. <break time="60s" /> Keep observing your breathing. <break time="60s" /> Keep observing your inner space. <break time="60s" /> <break time="60s" />

Stage 3 – Mental Thoughts Awareness — Entering the Breakthrough State
Now notice thoughts forming and dissolving. <break time="3s" /> Watch them as movements in the mind, like ripples on water. <break time="3s" /> Let each one pass without grasping. <break time="3s" /> Beneath them lies still awareness—open, spacious, calm. <break time="3s" /> This is the breakthrough: seeing thought without becoming it. <break time="3s" />
Stage Gap (5 minutes total): Focus on your breathing. <break time="60s" /> Focus on the sensations in the body. <break time="60s" /> Keep observing your breathing. <break time="60s" /> Keep observing your inner space. <break time="60s" /> <break time="60s" />

Stage 4 – Future Visualization — Positive Emotional Renewal
From this clear awareness, envision life unfolding with ease and kindness. <break time="3s" /> See yourself responding with balance and compassion. <break time="3s" /> Feel gratitude expanding quietly through the whole body. <break time="3s" /> Whisper inwardly: I am aware, I am balanced, I am free. <break time="3s" />

Final Stage Completion: <break time="180s" /> Move your feet. Move your toes. Slowly come back to your awareness.

Write in second person ("you") as if speaking directly to them during the meditation.""".strip()

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
