"""In-memory LRU + TTL response cache for deterministic, non-streaming requests."""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


def cache_key(model: str, body: dict[str, Any]) -> str:
    material = {
        "model": model,
        "messages": body.get("messages"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "max_tokens": body.get("max_tokens"),
        "stop": body.get("stop"),
        "response_format": body.get("response_format"),
        "tools": body.get("tools"),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


def cacheable(body: dict[str, Any]) -> bool:
    if body.get("stream"):
        return False
    if body.get("cache") is False:
        return False
    # Only cache deterministic requests unless the client opts in.
    return body.get("cache") is True or body.get("temperature", 1.0) == 0


class ResponseCache:
    def __init__(self, ttl_s: float, max_entries: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._data.get(key)
        if item is None:
            return None
        ts, value = item
        if self._clock() - ts > self.ttl_s:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        self._data[key] = (self._clock(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
