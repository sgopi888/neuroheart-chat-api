"""
Test multi-turn prompt to see what the model receives on turn 2.
Simulates: turn 1 with HRV, turn 2 without HRV (only chat history).
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

SYSTEM_PROMPT = (
    "You are NeuroHeart, a personal health insights assistant. "
    "Use the provided HRV and health context when relevant to give personalized advice. "
    "Keep answers concise, practical, and supportive."
)

HRV_CONTEXT = (
    "HRV_DAILY_14D (date: sdnn_ms, hr_bpm):\n"
    "2026-02-28: (51.55, 60.5), 2026-03-01: (59.87, 64.2), 2026-03-02: (60.46, 61.1), "
    "2026-03-03: (52.38, 74.1), 2026-03-04: (73.59, 91.4), 2026-03-05: (37.51, 97.1), "
    "2026-03-06: (45.38, 99.7), 2026-03-07: (53.48, 67.2), 2026-03-08: (49.08, 59.0), "
    "2026-03-09: (42.68, 95.5), 2026-03-10: (40.79, 95.5), 2026-03-11: (46.96, 85.9), "
    "2026-03-12: (47.68, 62.0), 2026-03-13: (56.01, 105.4)\n\n"
    "HEALTH_90D:\n"
    "hrv: mean_sdnn=48.91, trend=stable; hr: mean=76.5, p10=48.0, p90=124.0; "
    "sleep: mean_hours=0.6, trend=declining; steps: mean=12864.0, trend=improving"
)


def call(messages, label):
    print(f"\n{'='*60}")
    print(f"--- {label} ---")
    print(f"Messages: {len(messages)}")
    for m in messages:
        preview = m['content'][:80] + '...' if len(m['content']) > 80 else m['content']
        print(f"  [{m['role']}] {preview}")

    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=16384,
    )
    latency = time.time() - t0
    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    print(f"\nUsage: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}")
    print(f"Latency: {latency:.1f}s")
    print(f"Reply ({len(content)} chars):\n{content}\n")
    return content


# === TURN 1: With HRV context ===
turn1_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "system", "content": HRV_CONTEXT},
    {"role": "user", "content": "How is my HRV this week?"},
]
reply1 = call(turn1_messages, "TURN 1 — WITH HRV")

# === TURN 2A: WITH HRV again (current behavior — redundant) ===
turn2a_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "system", "content": HRV_CONTEXT},  # HRV injected again
    {"role": "user", "content": "How is my HRV this week?"},
    {"role": "assistant", "content": reply1},
    {"role": "user", "content": "What about my sleep trends? And which day had my best HRV?"},
]
reply2a = call(turn2a_messages, "TURN 2A — WITH HRV (current, redundant)")

# === TURN 2B: WITHOUT HRV (proposed — skip on follow-up) ===
turn2b_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    # NO HRV context — only chat history
    {"role": "user", "content": "How is my HRV this week?"},
    {"role": "assistant", "content": reply1},
    {"role": "user", "content": "What about my sleep trends? And which day had my best HRV?"},
]
reply2b = call(turn2b_messages, "TURN 2B — WITHOUT HRV (proposed)")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"\n2A (with HRV):    {len(reply2a)} chars")
print(f"2B (without HRV): {len(reply2b)} chars")
