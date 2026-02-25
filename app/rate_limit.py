from __future__ import annotations

import time
from typing import Dict, Tuple

# In-memory token bucket per user_uid.
# 20 requests per minute = capacity 20, refill rate 20/60 per second.
_buckets: Dict[str, Tuple[float, float]] = {}


def allow(
    user_uid: str,
    capacity: float = 20.0,
    refill_per_sec: float = 20.0 / 60.0,
) -> bool:
    now = time.time()
    tokens, last_ts = _buckets.get(user_uid, (capacity, now))
    tokens = min(capacity, tokens + (now - last_ts) * refill_per_sec)
    if tokens < 1.0:
        _buckets[user_uid] = (tokens, now)
        return False
    _buckets[user_uid] = (tokens - 1.0, now)
    return True
