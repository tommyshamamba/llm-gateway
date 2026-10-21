"""Per-key token-bucket rate limiter."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def check(self, key: str, rpm: int, burst: int) -> tuple[bool, float]:
        """Consume one token. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        rate = rpm / 60.0
        b = self._buckets.get(key)
        if b is None:
            b = self._buckets[key] = _Bucket(tokens=float(burst), updated=now)
        b.tokens = min(float(burst), b.tokens + (now - b.updated) * rate)
        b.updated = now
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - b.tokens) / rate if rate > 0 else 60.0
