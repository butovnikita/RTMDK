"""Tests for rtmdk.production.health_monitor."""

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.health_monitor import HealthMonitor


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class TestHealthMonitor:
    def test_check_health_healthy_empty(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        monitor = HealthMonitor(mem)
        health = monitor.check_health()
        assert health["status"] == "healthy"
        assert health["checks"]["node_count"]["value"] == 0

    def test_check_health_with_nodes(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(5):
            mem.save_context({"input": f"q{i}", "session_id": "s1"}, {"output": ""})
        monitor = HealthMonitor(mem)
        health = monitor.check_health()
        assert health["checks"]["node_count"]["value"] == 5

    def test_record_latency_and_degraded(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        monitor = HealthMonitor(mem)
        for _ in range(10):
            monitor.record_latency(600.0)
        health = monitor.check_health()
        assert health["status"] == "degraded"
        assert health["checks"]["latency"]["avg_ms"] == pytest.approx(600.0, 0.1)

    def test_add_alert_and_fire(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        monitor = HealthMonitor(mem)
        fired = []
        monitor.add_alert(
            "test_alert", threshold=0.5, callback=lambda name, status, checks: fired.append((name, status))
        )
        # Trigger degraded state with high latency
        monitor.record_latency(1000.0)
        monitor.check_health()
        assert len(fired) >= 1
        assert fired[0][0] == "test_alert"

    def test_get_metrics(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": ""})
        monitor = HealthMonitor(mem)
        monitor.check_health()
        metrics = monitor.get_metrics()
        assert metrics["rtmdk_nodes_total"] == 1
        assert metrics["rtmdk_health_status"] == "healthy"

    def test_get_metrics_text(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        monitor = HealthMonitor(mem)
        text = monitor.get_metrics_text()
        assert "rtmdk_nodes_total" in text
        assert "gauge" in text
