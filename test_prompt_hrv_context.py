#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import types


def _install_token_budget_stub() -> None:
    module = types.ModuleType("app.token_budget")
    module.MAX_TOKENS = 100_000
    module.count_tokens = lambda text: len((text or "").split())
    module.trim_text_to_tokens = lambda text, _max_tok: text
    sys.modules["app.token_budget"] = module


def main() -> int:
    _install_token_budget_stub()

    from app.chat_service import _build_prompt

    hrv_context = {
        "daily_14d": [
            {"date": "2026-03-12", "sdnn": 47.68, "mean_hr": 62.1},
            {"date": "2026-03-13", "sdnn": 56.01, "mean_hr": 55.0},
        ],
        "hrv_timeseries_14d": [
            {"timestamp": "2026-03-12T10:12:00+00:00", "value": 40.1},
            {"timestamp": "2026-03-13T08:12:22+00:00", "value": 30.67},
        ],
        "hrv_sdnn_timeseries_14d": [
            {"timestamp": "2026-03-12T11:02:00+00:00", "value": 46.9},
            {"timestamp": "2026-03-13T09:17:00+00:00", "value": 48.4},
        ],
        "hrv_daily_hourly_30d": [
            {"date": "2026-03-12", "window": "10:00-12:00", "avg_value": 40.1, "samples": 2},
            {"date": "2026-03-13", "window": "08:00-10:00", "avg_value": 30.67, "samples": 1},
        ],
        "hrv_sdnn_daily_hourly_30d": [
            {"date": "2026-03-12", "window": "10:00-12:00", "avg_value": 46.9, "samples": 1},
            {"date": "2026-03-13", "window": "08:00-10:00", "avg_value": 48.4, "samples": 1},
        ],
        "hrv_90d": {"mean_sdnn": 48.2, "trend": "stable"},
        "hrv_sdnn_90d": {
            "session_count": 12,
            "payload_count": 3,
            "value_count": 12,
            "mean_sdnn": 47.8,
            "trend": "stable",
        },
        "hr_90d": {"mean": 63.2, "p10": 52.0, "p90": 79.0},
        "sleep_90d": {"mean_hours": 7.1, "trend": "stable"},
        "steps_90d": {"mean": 8234, "trend": "improving"},
    }

    messages, breakdown = _build_prompt(
        summary="",
        history=[],
        hrv_context=hrv_context,
        rag_hits=[],
        user_message="What does my Apple Health data show?",
    )

    print("Prompt HRV blocks:")
    print("=" * 80)
    for message in messages:
        content = message.get("content", "")
        if (
            "HRV_DAILY_14D" in content
            or "HRV_TIMESERIES_14D" in content
            or "HRV_SDNN_TIMESERIES_14D" in content
            or "HRV_DAILY_HOURLY_30D" in content
            or "HRV_SDNN_DAILY_HOURLY_30D" in content
            or "HRV_AGGREGATES_90D" in content
        ):
            print(content)
            print("-" * 80)

    print("Aggregate keys present:")
    agg = {
        k: hrv_context[k]
        for k in ("hrv_90d", "hrv_sdnn_90d", "hr_90d", "sleep_90d", "steps_90d")
        if k in hrv_context
    }
    print(json.dumps(list(agg.keys())))
    print("Token breakdown:")
    print(json.dumps(breakdown, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
