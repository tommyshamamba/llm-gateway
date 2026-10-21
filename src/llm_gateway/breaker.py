"""Per-provider circuit breaker (closed -> open -> half-open)."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import IntEnum


class State(IntEnum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout_s: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._failures = 0
        self._opened_at = 0.0
        self._state = State.CLOSED
        self._probe_in_flight = False

    @property
    def state(self) -> State:
        if self._state is State.OPEN and self._clock() - self._opened_at >= self.reset_timeout_s:
            self._state = State.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def allow(self) -> bool:
        s = self.state
        if s is State.CLOSED:
            return True
        if s is State.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True  # single trial request
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._state = State.CLOSED
        self._probe_in_flight = False

    def record_failure(self) -> None:
        if self.state is State.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = State.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False
