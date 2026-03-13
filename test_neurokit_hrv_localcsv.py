#!/usr/bin/env python3
"""
Local CSV validation for Apple HRV vs NeuroKit2 computation.

Reads an exported `health_samples.csv` file, extracts RR-capable samples when
available, computes HRV locally via `app.hrv_neurokit`, and compares those
results to nearby Apple `hrv` rows from the same CSV.

Usage:
    ./.venv/bin/python test_neurokit_hrv_localcsv.py
    ./.venv/bin/python test_neurokit_hrv_localcsv.py /path/to/health_samples.csv
    ./.venv/bin/python test_neurokit_hrv_localcsv.py /path/to/health_samples.csv USER_ID
    ./.venv/bin/python test_neurokit_hrv_localcsv.py /path/to/health_samples.csv USER_ID 2026-03-07 7
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Keep matplotlib/font caches writable during NeuroKit import.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

sys.path.insert(0, os.path.dirname(__file__))

from app.hrv_neurokit import _compute_rr_metrics  # noqa: E402

DEFAULT_CSV = Path(__file__).with_name("health_samples.csv")
DEFAULT_USER = "001992.2a11e991f8d249a886acf50cec6920eb.0239"


@dataclass
class SampleRow:
    sample_type: str
    user_id: str
    start_time: datetime
    value: float | None
    payload: dict[str, Any] | None


def _parse_dt(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _parse_float(raw: str) -> float | None:
    if raw in ("", "NaN", "nan", None):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_payload(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _load_rows(csv_path: Path, user_id: str) -> list[SampleRow]:
    rows: list[SampleRow] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("user_id") != user_id:
                continue
            start_time = row.get("start_time")
            sample_type = row.get("sample_type")
            if not start_time or not sample_type:
                continue
            rows.append(
                SampleRow(
                    sample_type=sample_type,
                    user_id=user_id,
                    start_time=_parse_dt(start_time),
                    value=_parse_float(row.get("value")),
                    payload=_parse_payload(row.get("payload", "")),
                )
            )
    rows.sort(key=lambda row: row.start_time)
    return rows


def _resolve_window(
    rows: list[SampleRow], start_date_arg: str | None, days_arg: str | None
) -> tuple[date, date, int]:
    if days_arg is None:
        window_days = 7
    else:
        window_days = max(1, int(days_arg))

    if start_date_arg is None:
        latest_date = max(row.start_time.date() for row in rows)
        start_date = latest_date - timedelta(days=window_days - 1)
    else:
        start_date = date.fromisoformat(start_date_arg)

    end_date = start_date + timedelta(days=window_days - 1)
    return start_date, end_date, window_days


def _filter_rows_by_date(
    rows: list[SampleRow], start_date: date, end_date: date
) -> list[SampleRow]:
    return [row for row in rows if start_date <= row.start_time.date() <= end_date]


def _rr_from_payload(row: SampleRow) -> list[float]:
    payload = row.payload or {}

    if row.sample_type == "heartbeat_series":
        rr_list = payload.get("rr_intervals") or []
        return [
            float(entry["rr_interval_ms"])
            for entry in rr_list
            if isinstance(entry, dict) and entry.get("rr_interval_ms") is not None
        ]

    if row.sample_type == "hrv_sdnn":
        bpm_list = payload.get("beat_to_beat_bpm") or []
        rr_values: list[float] = []
        for bpm in bpm_list:
            try:
                bpm_value = float(bpm)
            except (TypeError, ValueError):
                continue
            if bpm_value > 0:
                rr_values.append(60000.0 / bpm_value)
        return rr_values

    return []


def _nearest_apple_hrv(rr_row: SampleRow, apple_rows: list[SampleRow]) -> SampleRow | None:
    same_day = [
        row
        for row in apple_rows
        if row.start_time.date() == rr_row.start_time.date() and row.value is not None
    ]
    if not same_day:
        return None
    return min(
        same_day,
        key=lambda row: abs((row.start_time - rr_row.start_time).total_seconds()),
    )


def _daily_apple_avg(rr_row: SampleRow, apple_rows: list[SampleRow]) -> float | None:
    values = [
        row.value
        for row in apple_rows
        if row.start_time.date() == rr_row.start_time.date() and row.value is not None
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _nearest_heart_rate(target_row: SampleRow, heart_rate_rows: list[SampleRow]) -> SampleRow | None:
    same_day = [
        row
        for row in heart_rate_rows
        if row.start_time.date() == target_row.start_time.date() and row.value is not None
    ]
    if not same_day:
        return None
    return min(
        same_day,
        key=lambda row: abs((row.start_time - target_row.start_time).total_seconds()),
    )


def _print_daily_apple_hrv(apple_rows: list[SampleRow], start_date: date, window_days: int) -> None:
    print("\nApple HRV per day:")
    print(f"{'date':12s} {'samples':>8s} {'avg_hrv':>10s} {'min':>10s} {'max':>10s}")
    print("-" * 56)

    rows_by_day: dict[date, list[float]] = {}
    for row in apple_rows:
        if row.value is None:
            continue
        rows_by_day.setdefault(row.start_time.date(), []).append(row.value)

    for offset in range(window_days):
        current_day = start_date + timedelta(days=offset)
        values = rows_by_day.get(current_day, [])
        if not values:
            print(f"{current_day.isoformat():12s} {0:8d} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}")
            continue
        avg_value = sum(values) / len(values)
        print(
            f"{current_day.isoformat():12s} {len(values):8d} "
            f"{avg_value:10.2f} {min(values):10.2f} {max(values):10.2f}"
        )


def _print_2h_apple_timeseries(apple_rows: list[SampleRow], start_date: date, window_days: int) -> None:
    print("\nApple HRV 2-hour time series:")

    rows_by_day: dict[date, list[SampleRow]] = {}
    for row in apple_rows:
        rows_by_day.setdefault(row.start_time.date(), []).append(row)

    for offset in range(window_days):
        current_day = start_date + timedelta(days=offset)
        print(f"\n{current_day.isoformat()}")
        print(f"{'window':12s} {'samples':>8s} {'avg_hrv':>10s}")
        print("-" * 34)

        day_rows = rows_by_day.get(current_day, [])
        for hour in range(0, 24, 2):
            bucket_values = [
                row.value
                for row in day_rows
                if row.value is not None and hour <= row.start_time.hour < hour + 2
            ]
            label = f"{hour:02d}:00-{hour + 2:02d}:00"
            if not bucket_values:
                print(f"{label:12s} {0:8d} {'N/A':>10s}")
                continue
            avg_value = sum(bucket_values) / len(bucket_values)
            print(f"{label:12s} {len(bucket_values):8d} {avg_value:10.2f}")


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    user_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_USER
    start_date_arg = sys.argv[3] if len(sys.argv) > 3 else None
    days_arg = sys.argv[4] if len(sys.argv) > 4 else None

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    rows = _load_rows(csv_path, user_id)
    if not rows:
        print(f"No rows found for user: {user_id}")
        return 1

    start_date, end_date, window_days = _resolve_window(rows, start_date_arg, days_arg)
    rows = _filter_rows_by_date(rows, start_date, end_date)
    if not rows:
        print(
            f"No rows found for user {user_id} in date range "
            f"{start_date.isoformat()} to {end_date.isoformat()}"
        )
        return 1

    counts = Counter(row.sample_type for row in rows)
    apple_rows = [row for row in rows if row.sample_type == "hrv" and row.value is not None]
    heart_rate_rows = [row for row in rows if row.sample_type == "heart_rate" and row.value is not None]
    rr_rows = [row for row in rows if row.sample_type in {"heartbeat_series", "hrv_sdnn"}]
    heartbeat_payload_rows = sum(
        1 for row in rows if row.sample_type == "heartbeat_series" and row.payload
    )
    hrv_sdnn_payload_rows = sum(
        1 for row in rows if row.sample_type == "hrv_sdnn" and row.payload
    )

    print(f"CSV: {csv_path}")
    print(f"User: {user_id}")
    print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()} ({window_days} days)")
    print(f"Rows loaded: {len(rows)}")
    print("Sample counts:")
    for sample_type, count in sorted(counts.items()):
        print(f"  {sample_type:20s} {count:>8,}")

    print("\nBPM / HRV source check:")
    print(f"  heart_rate.value rows (scalar BPM): {len(heart_rate_rows)}")
    print(f"  hrv.value rows (Apple HRV/SDNN, not BPM): {len(apple_rows)}")
    print(f"  hrv_sdnn payload rows (beat_to_beat_bpm): {hrv_sdnn_payload_rows}")
    print(f"  heartbeat_series payload rows (rr_intervals): {heartbeat_payload_rows}")
    print(
        "  NeuroKit2 needs beat-to-beat RR intervals or beat-to-beat BPM. "
        "heart_rate.value can show nearby BPM, but it cannot replace RR data."
    )

    _print_daily_apple_hrv(apple_rows, start_date, window_days)
    _print_2h_apple_timeseries(apple_rows, start_date, window_days)

    valid_sessions = 0
    insufficient_sessions = 0

    print("\nNeuroKit2 comparison candidates:")
    print(
        f"{'timestamp':25s} {'source':16s} {'rr_n':>6s} {'nk_sdnn':>10s} "
        f"{'apple_nearest':>14s} {'hr_nearest_bpm':>14s} {'apple_daily_avg':>15s} {'delta_nearest':>14s}"
    )
    print("-" * 126)

    for row in rr_rows:
        rr_values = _rr_from_payload(row)
        if not rr_values:
            continue

        nearest_apple = _nearest_apple_hrv(row, apple_rows)
        nearest_hr = _nearest_heart_rate(row, heart_rate_rows)
        daily_avg = _daily_apple_avg(row, apple_rows)

        if len(rr_values) < 10:
            insufficient_sessions += 1
            nearest_value = nearest_apple.value if nearest_apple else None
            nearest_hr_value = nearest_hr.value if nearest_hr else None
            print(
                f"{row.start_time.isoformat():25s} {row.sample_type:16s} {len(rr_values):6d} "
                f"{'INSUFFICIENT':>10s} "
                f"{f'{nearest_value:.2f}' if nearest_value is not None else 'N/A':>14s} "
                f"{f'{nearest_hr_value:.2f}' if nearest_hr_value is not None else 'N/A':>14s} "
                f"{f'{daily_avg:.2f}' if daily_avg is not None else 'N/A':>15s} "
                f"{'N/A':>14s}"
            )
            continue

        metrics = _compute_rr_metrics(rr_values)
        nk_sdnn = metrics.get("sdnn")
        nearest_value = nearest_apple.value if nearest_apple else None
        nearest_hr_value = nearest_hr.value if nearest_hr else None
        delta_nearest = None
        if nk_sdnn is not None and nearest_value is not None:
            delta_nearest = round(nk_sdnn - nearest_value, 2)

        valid_sessions += 1
        print(
            f"{row.start_time.isoformat():25s} {row.sample_type:16s} {len(rr_values):6d} "
            f"{f'{nk_sdnn:.2f}' if nk_sdnn is not None else 'N/A':>10s} "
            f"{f'{nearest_value:.2f}' if nearest_value is not None else 'N/A':>14s} "
            f"{f'{nearest_hr_value:.2f}' if nearest_hr_value is not None else 'N/A':>14s} "
            f"{f'{daily_avg:.2f}' if daily_avg is not None else 'N/A':>15s} "
            f"{f'{delta_nearest:.2f}' if delta_nearest is not None else 'N/A':>14s}"
        )

    print("\nSummary:")
    print(f"  Apple HRV rows: {len(apple_rows)}")
    print(f"  RR-capable rows: {len(rr_rows)}")
    print(f"  Valid NeuroKit sessions (>=10 RR): {valid_sessions}")
    print(f"  Insufficient RR sessions (<10 RR): {insufficient_sessions}")

    if valid_sessions == 0:
        print(
            "  No usable RR payloads were present in this CSV export, so there is no "
            "real NeuroKit-vs-iOS SDNN comparison to compute yet."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
