"""Tests for pipeline health monitor."""

from rtmdk.pipeline.health import PipelineHealthMonitor


class TestPipelineHealthMonitor:
    def test_set_threshold(self):
        monitor = PipelineHealthMonitor()
        monitor.set_threshold("embed", 100.0)
        assert monitor.thresholds["embed"] == 100.0

    def test_get_breaker_creates_new(self):
        monitor = PipelineHealthMonitor()
        monitor.set_threshold("embed", 100.0)
        breaker = monitor.get_breaker("embed")
        assert breaker.name == "embed"
        assert breaker.latency_threshold_ms == 100.0

    def test_get_breaker_returns_existing(self):
        monitor = PipelineHealthMonitor()
        b1 = monitor.get_breaker("embed")
        b2 = monitor.get_breaker("embed")
        assert b1 is b2

    def test_check_stage_healthy(self):
        monitor = PipelineHealthMonitor()
        monitor.set_threshold("embed", 100.0)
        assert monitor.check_stage("embed", 50.0, None) == "healthy"

    def test_check_stage_degraded(self):
        monitor = PipelineHealthMonitor()
        monitor.set_threshold("embed", 100.0)
        assert monitor.check_stage("embed", 150.0, None) == "degraded"

    def test_check_stage_failed(self):
        monitor = PipelineHealthMonitor()
        assert monitor.check_stage("embed", 50.0, "timeout") == "failed"

    def test_check_stage_default_threshold(self):
        monitor = PipelineHealthMonitor()
        # No threshold set, uses default 100.0
        assert monitor.check_stage("unknown", 50.0, None) == "healthy"
        assert monitor.check_stage("unknown", 150.0, None) == "degraded"

    def test_to_dict(self):
        monitor = PipelineHealthMonitor()
        monitor.get_breaker("embed")
        data = monitor.to_dict()
        assert "embed" in data
        assert "state" in data["embed"]
