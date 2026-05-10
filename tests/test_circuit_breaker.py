"""Unit tests for CircuitBreaker and AsyncCircuitBreaker."""

import time

import pytest

from rtmdk.support.circuit_breaker import CircuitBreaker, AsyncCircuitBreaker, CircuitState


class TestCircuitBreaker:
    def test_closed_state_allows_calls(self):
        cb = CircuitBreaker("test", failure_threshold=3, default="fallback")
        result = cb.call(lambda: "success")
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=2, default="fallback")
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 1
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN
        assert cb._failure_count == 2

    def test_open_returns_default_without_calling(self):
        cb = CircuitBreaker("test", failure_threshold=1, default="fallback")
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN
        calls = []
        result = cb.call(lambda: calls.append(1) or "success")
        assert result == "fallback"
        assert calls == []

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05, default="fallback")
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        result = cb.call(lambda: "recovery")
        assert result == "recovery"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.05, default="fallback")
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        # HALF_OPEN probe fails
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker("test", failure_threshold=3, default="fallback")
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb._failure_count == 1
        cb.call(lambda: "success")
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_exclude_exceptions_not_caught(self):
        cb = CircuitBreaker("test", failure_threshold=1, default="fallback")
        with pytest.raises(KeyboardInterrupt):
            cb.call(lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
        # Should not count as failure
        assert cb._failure_count == 0

    def test_get_state(self):
        cb = CircuitBreaker("test", failure_threshold=3, default="fallback")
        state = cb.get_state()
        assert state["name"] == "test"
        assert state["state"] == "closed"
        assert state["failure_count"] == 0


class TestAsyncCircuitBreaker:
    @pytest.mark.asyncio
    async def test_async_closed_state_allows_calls(self):
        cb = AsyncCircuitBreaker("async_test", failure_threshold=3, default="fallback")

        async def _success():
            return "success"

        result = await cb.call(_success)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_opens_after_failure_threshold(self):
        cb = AsyncCircuitBreaker("async_test", failure_threshold=2, default="fallback")

        async def fail():
            raise ValueError("fail")

        await cb.call(fail)
        assert cb.state == CircuitState.CLOSED
        await cb.call(fail)
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_async_open_returns_default(self):
        cb = AsyncCircuitBreaker("async_test", failure_threshold=1, default="fallback")

        async def fail():
            raise ValueError("fail")

        await cb.call(fail)
        assert cb.state == CircuitState.OPEN
        calls = []

        async def success():
            calls.append(1)
            return "success"

        result = await cb.call(success)
        assert result == "fallback"
        assert calls == []

    @pytest.mark.asyncio
    async def test_async_half_open_recovery(self):
        cb = AsyncCircuitBreaker("async_test", failure_threshold=1, recovery_timeout=0.05, default="fallback")

        async def fail():
            raise ValueError("fail")

        await cb.call(fail)
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)

        async def success():
            return "recovery"

        result = await cb.call(success)
        assert result == "recovery"
        assert cb.state == CircuitState.CLOSED
