"""Circuit breaker for pipeline stages.

Provides per-stage fault isolation: if a stage repeatedly fails or is too slow,
the breaker opens and the stage is automatically bypassed (fallback always runs).
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
import threading
import time


class BreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Bypassing stage, fallback always runs
    HALF_OPEN = "half_open"  # Testing if stage recovered


class CircuitBreaker:
    """Circuit breaker for a single pipeline stage.

    Opens after `failure_threshold` consecutive failures OR
    `latency_threshold_ms` exceeded `latency_violation_threshold` times.
    Attempts recovery after `recovery_timeout_ms`.
    """

    def __init__(
        self,
        name: str = "stage",
        failure_threshold: int = 5,
        latency_threshold_ms: float = 500.0,
        latency_violation_threshold: int = 3,
        recovery_timeout_ms: float = 30_000.0,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.latency_violation_threshold = latency_violation_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.half_open_max_calls = half_open_max_calls

        self._state = BreakerState.CLOSED
        self._failure_count = 0
        self._latency_violation_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._state

    def can_execute(self) -> bool:
        """Return True if the stage should execute (not bypassed)."""
        with self._lock:
            if self._state == BreakerState.CLOSED:
                return True
            if self._state == BreakerState.OPEN:
                if self._should_attempt_reset():
                    self._state = BreakerState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
                    return True
                return False
            # HALF_OPEN
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self, latency_ms: float) -> None:
        """Call after successful stage execution."""
        with self._lock:
            if latency_ms > self.latency_threshold_ms:
                self._latency_violation_count += 1
            else:
                self._latency_violation_count = max(0, self._latency_violation_count - 1)
            self._failure_count = max(0, self._failure_count - 1)
            if self._state == BreakerState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._reset()
                    return
            if (
                self._failure_count >= self.failure_threshold
                or self._latency_violation_count >= self.latency_violation_threshold
            ):
                self._state = BreakerState.OPEN

    def record_failure(self, latency_ms: float) -> None:
        """Call after failed stage execution or latency violation."""
        with self._lock:
            if latency_ms > self.latency_threshold_ms:
                self._latency_violation_count += 1
            self._failure_count += 1
            self._last_failure_time = time.perf_counter()

            if self._state == BreakerState.HALF_OPEN:
                self._state = BreakerState.OPEN
                return

            if (
                self._failure_count >= self.failure_threshold
                or self._latency_violation_count >= self.latency_violation_threshold
            ):
                self._state = BreakerState.OPEN

    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        elapsed = (time.perf_counter() - self._last_failure_time) * 1000
        return elapsed >= self.recovery_timeout_ms

    def _reset(self) -> None:
        self._state = BreakerState.CLOSED
        self._failure_count = 0
        self._latency_violation_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = None

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "latency_violation_count": self._latency_violation_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
            }
