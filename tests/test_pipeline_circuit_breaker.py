"""Tests for pipeline circuit breaker and SLO enforcement."""

import time
import pytest

from rtmdk.pipeline.base import PipelineContext, PipelineStage
from rtmdk.pipeline.circuit_breaker import CircuitBreaker, BreakerState
from rtmdk.pipeline.health import PipelineHealthMonitor


class FlakyStage(PipelineStage):
    """Stage that fails on first N calls, then succeeds."""
    name = "flaky"

    def __init__(self, fail_count: int = 3):
        self.fail_count = fail_count
        self.calls = 0

    def process(self, ctx: PipelineContext) -> PipelineContext:
        self.calls += 1
        if self.calls <= self.fail_count:
            raise RuntimeError(f"fail {self.calls}")
        ctx.results = [("ok", 1.0, None)]
        return ctx


class SlowStage(PipelineStage):
    """Stage that always takes a long time."""
    name = "slow"

    def process(self, ctx: PipelineContext) -> PipelineContext:
        time.sleep(0.02)  # 20ms
        ctx.results = [("ok", 1.0, None)]
        return ctx


class TestCircuitBreaker:
    def test_closed_allows_execution(self):
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        assert breaker.can_execute() is True
        assert breaker.state == BreakerState.CLOSED

    def test_opens_after_failures(self):
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        breaker.record_failure(1.0)
        breaker.record_failure(1.0)
        assert breaker.can_execute() is True  # still closed
        breaker.record_failure(1.0)
        assert breaker.state == BreakerState.OPEN
        assert breaker.can_execute() is False

    def test_opens_after_latency_violations(self):
        breaker = CircuitBreaker(
            name="test", latency_threshold_ms=10.0, latency_violation_threshold=2
        )
        breaker.record_success(5.0)   # ok
        breaker.record_success(15.0)  # violation 1
        assert breaker.state == BreakerState.CLOSED
        breaker.record_success(20.0)  # violation 2
        assert breaker.state == BreakerState.OPEN

    def test_half_open_then_close(self):
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout_ms=10.0)
        breaker.record_failure(1.0)
        breaker.record_failure(1.0)
        assert breaker.state == BreakerState.OPEN
        time.sleep(0.02)  # wait for recovery timeout
        assert breaker.can_execute() is True  # transitions to half-open
        assert breaker.state == BreakerState.HALF_OPEN
        breaker.record_success(1.0)
        breaker.record_success(1.0)
        breaker.record_success(1.0)
        assert breaker.state == BreakerState.CLOSED

    def test_half_open_then_reopen(self):
        breaker = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout_ms=10.0)
        breaker.record_failure(1.0)
        breaker.record_failure(1.0)
        time.sleep(0.02)
        breaker.can_execute()  # half-open
        breaker.record_failure(1.0)
        assert breaker.state == BreakerState.OPEN

    def test_stage_with_breaker_bypasses_when_open(self):
        stage = FlakyStage(fail_count=5)
        stage.circuit_breaker = CircuitBreaker(name="flaky", failure_threshold=2)

        # First 2 calls fail, breaker opens on 2nd failure
        ctx = PipelineContext(query_text="q")
        ctx = stage.run(ctx)
        assert ctx.metrics[-1].error is not None

        ctx = PipelineContext(query_text="q")
        ctx = stage.run(ctx)
        assert ctx.metrics[-1].error is not None
        assert stage.circuit_breaker.state == BreakerState.OPEN

        # 3rd call bypassed due to open breaker
        ctx = PipelineContext(query_text="q")
        ctx = stage.run(ctx)
        assert ctx.metrics[-1].error == "circuit_breaker_open"
        assert len(ctx.results) == 0  # fallback returned empty

    def test_to_dict(self):
        breaker = CircuitBreaker(name="x", failure_threshold=3)
        breaker.record_failure(1.0)
        d = breaker.to_dict()
        assert d["name"] == "x"
        assert d["state"] == "closed"
        assert d["failure_count"] == 1


class TestPipelineHealthMonitor:
    def test_set_threshold(self):
        monitor = PipelineHealthMonitor()
        monitor.set_threshold("embed", 100.0)
        assert monitor.thresholds["embed"] == 100.0

    def test_get_breaker_creates_new(self):
        monitor = PipelineHealthMonitor()
        monitor.set_threshold("embed", 100.0)
        b1 = monitor.get_breaker("embed")
        b2 = monitor.get_breaker("embed")
        assert b1 is b2
        assert b1.latency_threshold_ms == 100.0

    def test_check_stage(self):
        monitor = PipelineHealthMonitor()
        monitor.set_threshold("embed", 100.0)
        assert monitor.check_stage("embed", 50.0, None) == "healthy"
        assert monitor.check_stage("embed", 150.0, None) == "degraded"
        assert monitor.check_stage("embed", 50.0, "boom") == "failed"

    def test_to_dict(self):
        monitor = PipelineHealthMonitor()
        monitor.get_breaker("embed")
        d = monitor.to_dict()
        assert "embed" in d
        assert d["embed"]["state"] == "closed"


class TestConfigDrivenBreakers:
    def test_build_pipeline_reads_config(self):
        from rtmdk.memory.core import RTMDKMemory
        from rtmdk.memory.config import RTMDKConfig

        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=5,
            pipeline_breaker_enabled=True,
            pipeline_breaker_failure_threshold=2,
            pipeline_breaker_thresholds={"embed": 10.0, "retrieve": 10.0},
        )

        def embed(text):
            h = hash(text) % (2 ** 32)
            rng = np.random.default_rng(h)
            return rng.standard_normal(64, dtype=np.float32)

        mem = RTMDKMemory(config=cfg, embedder=embed)
        pipeline = mem.build_pipeline()
        for stage in pipeline.stages:
            assert stage.circuit_breaker is not None
            assert stage.circuit_breaker.failure_threshold == 2
            assert stage.name in ("embed", "route", "retrieve", "rerank", "calibrate", "explain")

    def test_build_pipeline_can_disable_breakers(self):
        from rtmdk.memory.core import RTMDKMemory
        from rtmdk.memory.config import RTMDKConfig

        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=5,
            pipeline_breaker_enabled=False,
        )

        def embed(text):
            h = hash(text) % (2 ** 32)
            rng = np.random.default_rng(h)
            return rng.standard_normal(64, dtype=np.float32)

        mem = RTMDKMemory(config=cfg, embedder=embed)
        pipeline = mem.build_pipeline()
        for stage in pipeline.stages:
            assert stage.circuit_breaker is None


class TestPipelineContextBreakerStates:
    def test_breaker_states_in_to_dict(self):
        stage = SlowStage()
        stage.circuit_breaker = CircuitBreaker(
            name="slow", latency_threshold_ms=5.0, latency_violation_threshold=1
        )
        ctx = PipelineContext(query_text="q")
        ctx = stage.run(ctx)
        result = ctx.to_dict()
        assert "breaker_states" in result
        assert result["breaker_states"]["slow"] == "open"  # 20ms > 5ms, 1 violation >= threshold=1

    def test_breaker_states_tracked_per_stage(self):
        stage1 = SlowStage()
        stage1.name = "s1"
        stage1.circuit_breaker = CircuitBreaker(name="s1", latency_threshold_ms=5.0, latency_violation_threshold=1)
        stage2 = SlowStage()
        stage2.name = "s2"
        stage2.circuit_breaker = CircuitBreaker(name="s2", latency_threshold_ms=50.0, latency_violation_threshold=1)

        ctx = PipelineContext(query_text="q")
        ctx = stage1.run(ctx)
        ctx = stage2.run(ctx)

        assert ctx.breaker_states["s1"] == "open"   # 20ms > 5ms, 1 violation >= threshold=1
        assert ctx.breaker_states["s2"] == "closed"  # 20ms < 50ms, no violation
