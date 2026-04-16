from __future__ import annotations

import time
from typing import Dict, Tuple

from app.config import settings

_buckets: Dict[str, Tuple[float, float]] = {}


def allow(
    user_uid: str,
    capacity: float = settings.rate_limit_capacity,
    refill_per_sec: float = settings.rate_limit_refill_per_sec,
) -> bool:
    now = time.time()
    tokens, last_ts = _buckets.get(user_uid, (capacity, now))
    tokens = min(capacity, tokens + (now - last_ts) * refill_per_sec)
    if tokens < 1.0:
        _buckets[user_uid] = (tokens, now)
        return False
    _buckets[user_uid] = (tokens - 1.0, now)
    return True
