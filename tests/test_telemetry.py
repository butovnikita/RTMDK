"""Tests for OpenTelemetry telemetry manager."""

from rtmdk.production.telemetry import TelemetryManager


class TestTelemetryManager:
    def test_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        mgr = TelemetryManager()
        assert not mgr.enabled

    def test_enabled_with_env(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        mgr = TelemetryManager()
        assert mgr.enabled
        mgr.shutdown()
