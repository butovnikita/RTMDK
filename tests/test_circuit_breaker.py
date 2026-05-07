"""Tests for rtmdk.support.circuit_breaker."""
import pytest
from rtmdk.support.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerBasics:
    def test_call_passes_through_on_success(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == CircuitState.CLOSED

    def test_call_returns_default_on_failure(self):
        cb = CircuitBreaker("test", failure_threshold=3, default="fallback")

        def fail():
            raise RuntimeError("boom")

        result = cb.call(fail)
        assert result == "fallback"
        assert cb._failure_count == 1

    def test_opens_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=2, default="fb")

        def fail():
            raise ValueError("x")

        cb.call(fail)
        cb.call(fail)
        assert cb.state == CircuitState.OPEN
        # Third call should fast-fail without invoking function
        result = cb.call(fail)
        assert result == "fb"
        assert cb._failure_count == 2  # no new failure counted while OPEN

    def test_excluded_exceptions_are_raised(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        with pytest.raises(KeyboardInterrupt):
            cb.call(lambda: (_ for _ in ()).throw(KeyboardInterrupt))

    def test_recovery_half_open_then_closed(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.0,
            default=0)

        def fail():
            raise RuntimeError("boom")

        cb.call(fail)
        assert cb.state == CircuitState.OPEN
        # Immediate retry (recovery_timeout=0) goes HALF_OPEN
        result = cb.call(lambda: 99)
        assert result == 99
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=1,
            recovery_timeout=0.0,
            default=0)

        def fail():
            raise RuntimeError("boom")

        cb.call(fail)
        assert cb.state == CircuitState.OPEN
        cb.call(fail)  # HALF_OPEN probe fails
        assert cb.state == CircuitState.OPEN

    def test_get_state(self):
        cb = CircuitBreaker("demo", failure_threshold=5)
        state = cb.get_state()
        assert state["name"] == "demo"
        assert state["state"] == "closed"
        assert state["failure_count"] == 0


class TestCircuitBreakerWithArgs:
    def test_call_forwards_args_and_kwargs(self):
        cb = CircuitBreaker("math")
        result = cb.call(lambda x, y, z=0: x + y + z, 1, y=2, z=3)
        assert result == 6
