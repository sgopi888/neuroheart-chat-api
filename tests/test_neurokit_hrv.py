#!/usr/bin/env python3
"""
Developer validation script: compare Apple HRV vs NeuroKit2 computation.

Usage (on VPS or locally with DB access):
    python test_neurokit_hrv.py [user_id]

If user_id is omitted, uses the default test user.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure app package is importable
sys.path.insert(0, os.path.dirname(__file__))

from app.db import get_engine  # noqa: E402
from app.hrv_neurokit import (  # noqa: E402
    _compute_rr_metrics,
    _detect_tier,
    fetch_hrv_context_local,
)
from sqlalchemy import text  # noqa: E402

DEFAULT_USER = "001992.2a11e991f8d249a886acf50cec6920eb.0239"


def show_tier_info(user_uid: str) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        tier = _detect_tier(conn, user_uid)
        print(f"\n{'='*60}")
        print(f"User: {user_uid}")
        print(f"Data Tier: {tier}")
        tier_desc = {1: "heartbeat_series (RR intervals)", 2: "hrv_sdnn (BPM payload)", 3: "Apple SDNN only"}
        print(f"Description: {tier_desc.get(tier, 'unknown')}")

        # Sample counts
        row = conn.execute(
            text("""
                SELECT sample_type, COUNT(*) AS cnt
                FROM health_samples
                WHERE user_id = :uid
                GROUP BY sample_type
                ORDER BY sample_type
            """),
            {"uid": user_uid},
        ).fetchall()
        print(f"\nSample counts:")
        for r in row:
            print(f"  {r.sample_type:20s} {r.cnt:>8,}")
        print(f"{'='*60}")


def show_apple_sdnn_daily(user_uid: str) -> None:
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT DATE(start_time AT TIME ZONE 'UTC') AS day,
                       AVG(value) AS avg_sdnn,
                       MIN(value) AS min_sdnn,
                       MAX(value) AS max_sdnn,
                       COUNT(*) AS samples
                FROM health_samples
                WHERE user_id = :uid
                  AND sample_type = 'hrv'
                  AND start_time >= NOW() - INTERVAL '14 days'
                  AND value IS NOT NULL
                GROUP BY DATE(start_time AT TIME ZONE 'UTC')
                ORDER BY day ASC
            """),
            {"uid": user_uid},
        ).fetchall()

    print(f"\nApple HRV SDNN — last 14 days:")
    print(f"{'date':>12} {'avg_sdnn':>10} {'min':>8} {'max':>8} {'samples':>8}")
    print("-" * 52)
    for r in rows:
        print(f"{str(r.day):>12} {r.avg_sdnn:>10.2f} {r.min_sdnn:>8.2f} {r.max_sdnn:>8.2f} {r.samples:>8}")


def show_neurokit_comparison(user_uid: str) -> None:
    """If RR data exists, compare Apple vs NeuroKit SDNN."""
    eng = get_engine()
    with eng.begin() as conn:
        # Check for heartbeat_series
        rows = conn.execute(
            text("""
                SELECT DATE(start_time AT TIME ZONE 'UTC') AS day, payload
                FROM health_samples
                WHERE user_id = :uid AND sample_type = 'heartbeat_series'
                  AND payload IS NOT NULL
                  AND start_time >= NOW() - INTERVAL '14 days'
                ORDER BY start_time ASC
            """),
            {"uid": user_uid},
        ).fetchall()

    if not rows:
        print("\nNo heartbeat_series data yet — NeuroKit comparison unavailable.")
        print("(Once iOS sends heartbeat_series samples, this script will show Apple vs NeuroKit SDNN)")
        return

    import json

    print(f"\nApple vs NeuroKit2 SDNN comparison:")
    print(f"{'date':>12} {'apple_sdnn':>12} {'nk_sdnn':>10} {'diff%':>8} {'rmssd':>8} {'lf_hf':>8}")
    print("-" * 62)

    for r in rows:
        payload = r.payload if isinstance(r.payload, dict) else json.loads(r.payload)
        rr_list = payload.get("rr_intervals", [])
        rr = [e["rr_interval_ms"] for e in rr_list if isinstance(e, dict) and "rr_interval_ms" in e]
        if len(rr) < 10:
            continue
        metrics = _compute_rr_metrics(rr)
        nk_sdnn = metrics.get("sdnn")
        rmssd = metrics.get("rmssd")
        lf_hf = metrics.get("lf_hf_ratio")

        # No Apple comparison since these are RR-based
        print(
            f"{str(r.day):>12} {'N/A':>12} "
            f"{nk_sdnn if nk_sdnn else 'N/A':>10} "
            f"{'N/A':>8} "
            f"{rmssd if rmssd else 'N/A':>8} "
            f"{lf_hf if lf_hf else 'N/A':>8}"
        )


async def show_full_context(user_uid: str) -> None:
    """Show the complete HRV context that would be sent to the LLM."""
    import json

    for mode in ("compact", "meditation", "full"):
        ctx = await fetch_hrv_context_local(user_uid, "7d", mode=mode)
        print(f"\n--- HRV Context (mode={mode}) ---")
        print(json.dumps(ctx, indent=2, default=str))


def main():
    user_uid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER

    show_tier_info(user_uid)
    show_apple_sdnn_daily(user_uid)
    show_neurokit_comparison(user_uid)
    asyncio.run(show_full_context(user_uid))


if __name__ == "__main__":
    main()
