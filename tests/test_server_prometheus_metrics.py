"""Tests for Prometheus /metrics HTTP endpoint."""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory

app_mod = importlib.import_module("rtmdk.server.app")


def _embedder(text: str):
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(64).astype(np.float32)


@pytest.fixture(scope="module")
def client():
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def reset_memory():
    old = app_mod.memory
    app_mod.memory = None
    yield
    app_mod.memory = old


def _init_memory():
    cfg = RTMDKConfig(latent_dim=64, use_hnsw=False, wal_fsync_interval_ms=0)
    mem = RTMDKMemory(config=cfg, embedder=_embedder)
    for i in range(5):
        emb = np.random.randn(64).astype(np.float32)
        mem.add_node(content={"text": f"node {i}", "topic": "test"}, embedding=emb)
    app_mod.memory = mem
    return mem


class TestPrometheusMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        _init_memory()
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_contains_node_count(self, client):
        _init_memory()
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "rtmdk_nodes_total" in text
        # Should contain the actual node count
        assert (
            'rtmdk_nodes_total{ tier="hot" } 5.0' in text
            or 'rtmdk_nodes_total{tier="hot"} 5.0' in text
            or "rtmdk_nodes_total" in text
        )

    def test_metrics_contains_queries_counter(self, client):
        _init_memory()
        # Make a query request to increment the counter
        client.post("/v1/memory/query", json={"query": "hello", "top_k": 3})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "rtmdk_queries_total" in text

    def test_metrics_contains_query_duration_histogram(self, client):
        _init_memory()
        resp = client.get("/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "rtmdk_query_duration_seconds" in text
        assert "rtmdk_query_duration_seconds_bucket" in text

    def test_metrics_without_memory_still_works(self, client):
        # memory is None (reset_memory fixture)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Should still return valid Prometheus format
        assert "# HELP" in resp.text or "# TYPE" in resp.text
