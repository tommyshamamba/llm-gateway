"""Token and cost accounting per client key."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from .config import Pricing


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0


def estimate_tokens(text: str) -> int:
    """Rough fallback (~4 chars/token) when a provider returns no usage."""
    return max(1, len(text) // 4) if text else 0


def cost_usd(pricing: Pricing | None, prompt_tokens: int, completion_tokens: int) -> float:
    if pricing is None:
        return 0.0
    return (prompt_tokens * pricing.input + completion_tokens * pricing.output) / 1_000_000


class Ledger:
    def __init__(self) -> None:
        self._by_key: dict[str, Usage] = defaultdict(Usage)

    def record(self, key: str, prompt: int, completion: int, cost: float, cache_hit: bool = False) -> None:
        u = self._by_key[key]
        u.requests += 1
        u.prompt_tokens += prompt
        u.completion_tokens += completion
        u.cost_usd += cost
        u.cache_hits += int(cache_hit)

    def snapshot(self, key: str | None = None) -> dict:
        if key is not None:
            return asdict(self._by_key[key])
        return {k: asdict(v) for k, v in self._by_key.items()}
