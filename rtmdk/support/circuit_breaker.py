"""Circuit breaker pattern for RTMDK subsystem fault tolerance — R7.1 canonical.

R7.1 (2026-08-24, audit/risks-2026-08-24): single source for 3-state
CLOSED/OPEN/HALF_OPEN (CircuitState) + base CircuitBreaker (failure_threshold,
recovery_timeout, call wrapper). Pipeline's latency-aware breaker
(pipeline/circuit_breaker.py) re-uses this state enum (BreakerState alias)
and inherits thresholds from RTMDKConfig.pipeline_breaker_thresholds
(config.py:748) via pipeline_builder. Field 11 breakers, embedder, server
circuits all use this class — minimizes duplication (was duplicated with
pipeline's BreakerState).

Replaces the overly broad _safe_run() catch-all with per-subsystem
failure tracking and automatic recovery.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for a single subsystem.

    - CLOSED: calls pass through; failures are counted.
    - After *failure_threshold* consecutive failures → OPEN.
    - OPEN: calls return *default* immediately (no call).
    - After *recovery_timeout* seconds → HALF_OPEN.
    - HALF_OPEN: one probe call allowed.
        - Success → CLOSED.
        - Failure → OPEN again.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        default: Any = None,
        exceptions: tuple = (Exception,),
        exclude_exceptions: tuple = (KeyboardInterrupt, SystemExit),
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.default = default
        self.exceptions = exceptions
        self.exclude_exceptions = exclude_exceptions

        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Invoke *func* through the circuit breaker."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"[CircuitBreaker:{self.name}] HALF_OPEN — probing recovery")
            else:
                logger.debug(f"[CircuitBreaker:{self.name}] OPEN — fast-fail ({self._failure_count} failures)")
                return self.default  # type: ignore[return-value]

        try:
            result = func(*args, **kwargs)
        except self.exclude_exceptions:
            raise
        except self.exceptions as e:
            self._on_failure(e)
            return self.default  # type: ignore[return-value]

        self._on_success()
        return result

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"[CircuitBreaker:{self.name}] CLOSED — recovery confirmed")
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None

    def _on_failure(self, exc: Exception):
        self._failure_count += 1
        self._last_failure_time = time.time()
        logger.exception(f"[CircuitBreaker:{self.name}] failure {self._failure_count}/{self.failure_threshold}: {exc}")
        if self._failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"[CircuitBreaker:{self.name}] OPEN — threshold reached, fast-failing for {self.recovery_timeout}s"
            )

    def _should_attempt_reset(self) -> bool:
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.recovery_timeout

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
        }


class AsyncCircuitBreaker(CircuitBreaker):
    """Async-aware circuit breaker for coroutine functions."""

    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"[CircuitBreaker:{self.name}] HALF_OPEN — probing recovery")
            else:
                logger.debug(f"[CircuitBreaker:{self.name}] OPEN — fast-fail ({self._failure_count} failures)")
                return self.default

        try:
            result = await func(*args, **kwargs)
        except self.exclude_exceptions:
            raise
        except self.exceptions as e:
            self._on_failure(e)
            return self.default

        self._on_success()
        return result
