"""Failover engine: ordered targets per route, retries with jittered backoff,
and a circuit breaker per provider."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .breaker import CircuitBreaker
from .config import Settings, Target
from .logging import log
from .metrics import Metrics
from .providers import Provider, ProviderError

CLIENT_ERRORS = {400, 413, 422}  # caller's fault: do not fail over


class NoHealthyProvider(Exception):
    def __init__(self, route: str, errors: list[str]) -> None:
        super().__init__(f"all targets failed for {route}: {errors}")
        self.errors = errors


class ClientRequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Outcome:
    target: Target
    attempts: int
    failover: bool
    errors: list[str] = field(default_factory=list)


class Router:
    def __init__(self, settings: Settings, providers: dict[str, Provider], metrics: Metrics) -> None:
        self.settings = settings
        self.providers = providers
        self.metrics = metrics
        self.breakers = {
            name: CircuitBreaker(settings.breaker.failure_threshold, settings.breaker.reset_timeout_s)
            for name in providers
        }

    def targets(self, route: str) -> list[Target]:
        r = self.settings.routes.get(route)
        if r is None:
            raise KeyError(route)
        return r.targets

    def _gauge(self, name: str) -> None:
        self.metrics.breaker_state.labels(name).set(int(self.breakers[name].state))

    async def _backoff(self, attempt: int) -> None:
        rc = self.settings.retry
        delay = min(rc.backoff_max_s, rc.backoff_base_s * (2 ** attempt))
        await asyncio.sleep(delay * (0.5 + random.random() / 2))

    def _fail(self, name: str, err: ProviderError) -> None:
        self.breakers[name].record_failure()
        self.metrics.upstream.labels(name, "error").inc()
        self._gauge(name)
        if err.status in CLIENT_ERRORS:
            raise ClientRequestError(err.status or 400, str(err))

    def _ok(self, name: str) -> None:
        self.breakers[name].record_success()
        self.metrics.upstream.labels(name, "ok").inc()
        self._gauge(name)

    async def complete(self, route: str, body: dict[str, Any]) -> tuple[dict[str, Any], Outcome]:
        errors: list[str] = []
        attempts = 0
        for idx, t in enumerate(self.targets(route)):
            prov, br = self.providers[t.provider], self.breakers[t.provider]
            for attempt in range(self.settings.retry.attempts_per_provider):
                if not br.allow():
                    errors.append(f"{t.provider}: circuit open")
                    self.metrics.upstream.labels(t.provider, "skipped_open").inc()
                    break
                attempts += 1
                t0 = time.perf_counter()
                try:
                    resp = await prov.complete(t.model, body)
                except ProviderError as e:
                    self.metrics.upstream_latency.labels(t.provider).observe(time.perf_counter() - t0)
                    errors.append(str(e))
                    log.warning("upstream_error", extra={"provider": t.provider, "error": str(e),
                                                          "attempt": attempt + 1})
                    self._fail(t.provider, e)
                    if not e.retryable:
                        break
                    if attempt + 1 < self.settings.retry.attempts_per_provider:
                        await self._backoff(attempt)
                    continue
                self.metrics.upstream_latency.labels(t.provider).observe(time.perf_counter() - t0)
                self._ok(t.provider)
                return resp, Outcome(t, attempts, idx > 0, errors)
        raise NoHealthyProvider(route, errors)

    async def stream(
        self, route: str, body: dict[str, Any]
    ) -> tuple[AsyncIterator[dict[str, Any]], Outcome]:
        """Fail over until a target yields its first chunk; after that the stream
        is committed to that target (mid-stream errors are surfaced, not replayed)."""
        errors: list[str] = []
        attempts = 0
        for idx, t in enumerate(self.targets(route)):
            prov, br = self.providers[t.provider], self.breakers[t.provider]
            for attempt in range(self.settings.retry.attempts_per_provider):
                if not br.allow():
                    errors.append(f"{t.provider}: circuit open")
                    self.metrics.upstream.labels(t.provider, "skipped_open").inc()
                    break
                attempts += 1
                gen = prov.stream(t.model, body)
                try:
                    first = await gen.__anext__()
                except StopAsyncIteration:
                    first = None
                except ProviderError as e:
                    errors.append(str(e))
                    self._fail(t.provider, e)
                    if not e.retryable:
                        break
                    if attempt + 1 < self.settings.retry.attempts_per_provider:
                        await self._backoff(attempt)
                    continue
                self._ok(t.provider)
                return self._chain(first, gen, t.provider), Outcome(t, attempts, idx > 0, errors)
        raise NoHealthyProvider(route, errors)

    async def _chain(self, first, gen, name: str) -> AsyncIterator[dict[str, Any]]:
        if first is not None:
            yield first
        try:
            async for c in gen:
                yield c
        except ProviderError as e:
            self._fail(name, e)
            raise
