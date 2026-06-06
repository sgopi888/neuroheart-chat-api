"""Deterministic safety guardrails for the wellness chat.

NeuroHeart is positioned as a wellness, mindfulness, and lifestyle-coaching
companion — not a medical app. These guardrails run *before* the LLM call so
that, in clear emergency or crisis situations, the user is routed to real help
instead of receiving AI-generated coaching content.

Two checks are provided:

* ``detect_emergency`` — high-precision patterns for crisis / medical
  emergencies (suicide, self-harm, overdose, stroke, heart attack, etc.).
  When matched, the caller should return ``EMERGENCY_RESPONSE`` directly and
  must NOT generate coaching content.

* ``looks_out_of_scope`` — softer heuristic for medical (non-emergency)
  requests such as symptom interpretation or medication questions. This is a
  hint only; the system prompt already instructs the model to redirect these,
  so callers may choose to short-circuit or simply rely on the model.

Matching is intentionally conservative (word-boundary regexes, no broad
substring matches) to avoid false positives on ordinary wellness conversation
(e.g. "I want to kill this bad habit" must NOT trigger the emergency path).
"""

from __future__ import annotations

import re
from typing import List, Pattern

from app.prompts import EMERGENCY_RESPONSE, OUT_OF_SCOPE_RESPONSE

__all__ = [
    "EMERGENCY_RESPONSE",
    "OUT_OF_SCOPE_RESPONSE",
    "detect_emergency",
    "looks_out_of_scope",
]


def _compile(patterns: List[str]) -> List[Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# ── Emergency / crisis patterns ────────────────────────────────────
# Phrased to require intent/context, not just a sensitive keyword, so that
# everyday wellness language does not trip the guardrail.
_EMERGENCY_PATTERNS: List[Pattern[str]] = _compile([
    # Suicide / self-harm intent
    r"\bkill(?:ing)?\s+myself\b",
    r"\bend(?:ing)?\s+my\s+life\b",
    r"\btake\s+my\s+(?:own\s+)?life\b",
    r"\bwant\s+to\s+die\b",
    r"\bdon'?t\s+want\s+to\s+(?:live|be\s+alive)\b",
    r"\b(?:commit|committing)\s+suicide\b",
    r"\bsuicidal\b",
    r"\bself[\s-]?harm(?:ing)?\b",
    r"\bcut(?:ting)?\s+myself\b",
    r"\bhurt(?:ing)?\s+myself\b",
    r"\bharm(?:ing)?\s+myself\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bbetter\s+off\s+dead\b",

    # Overdose
    r"\boverdos(?:e|ed|ing)\b",
    r"\btook\s+too\s+many\s+(?:pills|tablets)\b",

    # Acute medical emergencies
    r"\bheart\s+attack\b",
    r"\bstroke\b",
    r"\bseizure\b",
    r"\bcan'?t\s+breathe\b",
    r"\bcannot\s+breathe\b",
    r"\bstopped\s+breathing\b",
    r"\bchest\s+pain\b",
    r"\bunconscious\b",
    r"\boverdosing\b",
    r"\bbleeding\s+(?:badly|heavily|a\s+lot|out)\b",
    r"\bemergency\s+room\b",
    r"\b911\b",
])


# ── Medical (non-emergency) out-of-scope hints ─────────────────────
_MEDICAL_PATTERNS: List[Pattern[str]] = _compile([
    r"\b(?:do|might)\s+i\s+have\b.*\b(?:covid|flu|cancer|diabetes|infection|disease)\b",
    r"\bdiagnos(?:e|is|ed|ing)\b",
    r"\bwhat\s+(?:medicine|medication|drug|pill|antibiotic)s?\s+(?:should|can|do)\b",
    r"\b(?:dosage|dose)\s+of\b",
    r"\bhow\s+much\s+\w+\s+should\s+i\s+take\b",
    r"\bi\s+have\s+(?:a\s+)?fever\b",
    r"\bmy\s+(?:temperature|fever)\s+is\b",
    r"\bis\s+this\s+(?:a\s+)?symptom\b",
    r"\btreat(?:ment)?\s+(?:plan|for)\b.*\b(?:disease|condition|infection|illness)\b",
    r"\bprescri(?:be|ption)\b",
])


def detect_emergency(text: str) -> bool:
    """Return True if the message indicates a crisis or medical emergency.

    When True, the caller must return :data:`EMERGENCY_RESPONSE` and must NOT
    generate any coaching/wellness content for this turn.
    """
    if not text:
        return False
    return any(p.search(text) for p in _EMERGENCY_PATTERNS)


def looks_out_of_scope(text: str) -> bool:
    """Return True if the message looks like a (non-emergency) medical request.

    This is a soft hint. The system prompt already instructs the model to
    redirect these with :data:`OUT_OF_SCOPE_RESPONSE`; callers may use this to
    short-circuit deterministically if stricter enforcement is desired.
    """
    if not text:
        return False
    return any(p.search(text) for p in _MEDICAL_PATTERNS)
