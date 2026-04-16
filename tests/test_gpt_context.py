"""
Test GPT-5-nano context window limits.

Sends a system message padded with repeated health-context words
to see when the model stops responding.

Usage:
    python test_gpt_context.py
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Tunables ---
CONTEXT_WORDS = 250_000  # number of filler words in system context
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
MAX_COMPLETION_TOKENS = 2048

# Realistic filler: health/HRV keywords (deduped via repetition)
_FILLER_VOCAB = (
    "heart rate variability SDNN RMSSD HRV stress recovery sleep "
    "autonomic nervous system parasympathetic sympathetic vagal tone "
    "breathing meditation mindfulness relaxation exercise steps daily "
    "baseline trend improving declining stable circadian rhythm cortisol "
    "wellness resilience adaptive capacity coherence biofeedback wearable "
    "Apple Watch HealthKit health samples data insight personalized "
)

_FILLER_WORDS = _FILLER_VOCAB.split()


def build_filler(n_words: int) -> str:
    """Generate n_words of repeated health vocabulary."""
    full_repeats = n_words // len(_FILLER_WORDS)
    remainder = n_words % len(_FILLER_WORDS)
    parts = _FILLER_WORDS * full_repeats + _FILLER_WORDS[:remainder]
    return " ".join(parts)


def test_context(n_words: int) -> dict:
    """Send a prompt with n_words filler and return result info."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    filler = build_filler(n_words)
    word_count = len(filler.split())

    messages = [
        {
            "role": "system",
            "content": (
                "You are a health assistant. The following is health context data. "
                "After reading it, answer the user's question.\n\n"
                f"HEALTH_DATA:\n{filler}"
            ),
        },
        {
            "role": "user",
            "content": "Hello, how are you? Give me a brief health tip.",
        },
    ]

    print(f"\n{'='*60}")
    print(f"Testing with {word_count:,} filler words (~{word_count * 4 // 3:,} tokens)")
    print(f"Model: {MODEL}")

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
        latency = time.time() - t0
        choice = resp.choices[0]
        content = (choice.message.content or "").strip()

        result = {
            "words": word_count,
            "est_tokens": word_count * 4 // 3,
            "finish_reason": choice.finish_reason,
            "content_len": len(content),
            "content_preview": content[:200] if content else "(empty)",
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                "total_tokens": resp.usage.total_tokens if resp.usage else None,
            },
            "latency_s": round(latency, 2),
            "error": None,
        }
    except Exception as exc:
        latency = time.time() - t0
        result = {
            "words": word_count,
            "est_tokens": word_count * 4 // 3,
            "finish_reason": None,
            "content_len": 0,
            "content_preview": None,
            "usage": None,
            "latency_s": round(latency, 2),
            "error": str(exc),
        }

    # Print results
    print(f"Latency: {result['latency_s']}s")
    if result["error"]:
        print(f"ERROR: {result['error']}")
    else:
        print(f"Finish reason: {result['finish_reason']}")
        print(f"Usage: {result['usage']}")
        print(f"Reply ({result['content_len']} chars): {result['content_preview']}")

    return result


if __name__ == "__main__":
    # Test at different context sizes
    sizes = [1_000, 50_000, 100_000, 150_000, 200_000, CONTEXT_WORDS]

    print(f"GPT Context Window Test — Model: {MODEL}")
    print(f"Max completion tokens: {MAX_COMPLETION_TOKENS}")

    results = []
    for n in sizes:
        r = test_context(n)
        results.append(r)

        # Stop if we got an error (likely context too large)
        if r["error"]:
            print(f"\nStopping — hit error at {n:,} words")
            break

        # Stop if empty response
        if r["content_len"] == 0:
            print(f"\nStopping — got empty response at {n:,} words")
            break

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Words':>10} {'Est Tok':>10} {'Prompt Tok':>10} {'Reply':>6} {'Finish':>12} {'Latency':>8}")
    for r in results:
        prompt_tok = r["usage"]["prompt_tokens"] if r["usage"] else "-"
        reply_len = r["content_len"]
        finish = r["finish_reason"] or "ERROR"
        print(f"{r['words']:>10,} {r['est_tokens']:>10,} {str(prompt_tok):>10} {reply_len:>6} {finish:>12} {r['latency_s']:>7.1f}s")
