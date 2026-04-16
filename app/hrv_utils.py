"""Shared HRV computation from RR intervals."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def compute_hrv_from_rr(
    rr_intervals: List[float], min_count: int = 5
) -> Optional[Dict[str, Any]]:
    if len(rr_intervals) < min_count:
        return None
    n = len(rr_intervals)
    mean_nn = sum(rr_intervals) / n
    variance = sum((x - mean_nn) ** 2 for x in rr_intervals) / max(n - 1, 1)
    sdnn = math.sqrt(variance)
    diffs = [rr_intervals[i + 1] - rr_intervals[i] for i in range(n - 1)]
    rmssd = math.sqrt(sum(d ** 2 for d in diffs) / max(len(diffs), 1))
    nn50 = sum(1 for d in diffs if abs(d) > 50)
    pnn50 = (nn50 / len(diffs)) * 100 if diffs else 0
    mean_hr = 60000.0 / mean_nn if mean_nn > 0 else None
    return {
        "sdnn": round(sdnn, 2),
        "rmssd": round(rmssd, 2),
        "pnn50": round(pnn50, 2),
        "mean_nn": round(mean_nn, 2),
        "mean_hr": round(mean_hr, 1) if mean_hr else None,
        "beat_count": n,
    }
