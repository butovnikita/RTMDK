"""Test Phase 4 observability components."""
import asyncio
import pytest

from rtmdk.support.circuit_breaker import AsyncCircuitBreaker, CircuitState
from rtmdk.production.rate_limiter import RateLimiter


class TestAsyncCircuitBreaker:
    @pytest.mark.asyncio
    async def test_async_circuit_allows_success(self):
        cb = AsyncCircuitBreaker("test", failure_threshold=2, recovery_timeout=1.0, default="fallback")
        async def ok():
            return "ok"
        result = await cb.call(ok)
        assert result == "ok"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_async_circuit_opens_after_failures(self):
        cb = AsyncCircuitBreaker("test", failure_threshold=2, recovery_timeout=60.0, default="fallback")

        async def fail():
            raise RuntimeError("boom")

        await cb.call(fail)
        await cb.call(fail)
        assert cb.state == CircuitState.OPEN

        async def ok():
            return "ok"
        result = await cb.call(ok)
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_async_circuit_half_open_recovery(self):
        cb = AsyncCircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1, default="fallback")

        async def fail():
            raise RuntimeError("boom")

        await cb.call(fail)
        assert cb.state == CircuitState.OPEN
        await asyncio.sleep(0.15)

        async def recovered():
            return "recovered"
        result = await cb.call(recovered)
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED


class TestRateLimiter:
    def test_allow_request_within_limits(self):
        rl = RateLimiter(max_per_minute=10, max_per_hour=100, max_per_day=1000)
        assert rl.allow_request("client1")
        assert rl.allow_request("client1")

    def test_blocks_when_exceeded(self):
        rl = RateLimiter(max_per_minute=1, max_per_hour=100, max_per_day=1000)
        assert rl.allow_request("client1")
        assert not rl.allow_request("client1")

    def test_remaining_counts(self):
        rl = RateLimiter(max_per_minute=5, max_per_hour=100, max_per_day=1000)
        rl.allow_request("client1")
        remaining = rl.get_remaining("client1")
        assert remaining["per_minute"] == 4
