"""Tests for /v1/memory/pipeline/prometheus endpoint."""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory

app_mod = importlib.import_module("rtmdk.server.app")


def _embedder(text: str):
    return np.array([0.0] * 16)


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


class TestPipelinePrometheus:
    def test_pipeline_prometheus_no_memory(self, client):
        resp = client.get("/v1/memory/pipeline/prometheus")
        assert resp.status_code == 503

    def test_pipeline_prometheus_with_memory(self, client):
        cfg = RTMDKConfig(
            latent_dim=16,
            use_hnsw=False,
            wal_fsync_interval_ms=0,
            pipeline_breaker_enabled=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        mem.add_node(content={"text": "hello"}, embedding=np.array([0.0] * 16))
        app_mod.memory = mem
        try:
            resp = client.get("/v1/memory/pipeline/prometheus")
            assert resp.status_code == 200
            text = resp.text
            assert "rtmdk_pipeline_stages_total" in text
            assert "rtmdk_pipeline_stage_enabled" in text
            assert "rtmdk_pipeline_breaker_state" in text
            assert "text/plain" in resp.headers.get("content-type", "")
        finally:
            app_mod.memory = None

    def test_pipeline_prometheus_with_metrics_store(self, client, tmp_path):
        from rtmdk.pipeline.persistence import PipelineMetricsStore

        cfg = RTMDKConfig(
            latent_dim=16,
            use_hnsw=False,
            wal_fsync_interval_ms=0,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        app_mod.memory = mem
        old_store = app_mod.pipeline_metrics_store
        app_mod.pipeline_metrics_store = PipelineMetricsStore(str(tmp_path / "metrics.jsonl"))
        app_mod.pipeline_metrics_store.write(
            {
                "query_text": "q1",
                "total_latency_ms": 10.0,
                "stages": [
                    {"stage": "embed", "latency_ms": 5.0, "error": None, "degraded": False},
                ],
            }
        )
        try:
            resp = client.get("/v1/memory/pipeline/prometheus")
            assert resp.status_code == 200
            text = resp.text
            assert "rtmdk_pipeline_queries_total" in text
            assert "rtmdk_pipeline_stage_latency_ms" in text
        finally:
            app_mod.pipeline_metrics_store = old_store
            app_mod.memory = None
