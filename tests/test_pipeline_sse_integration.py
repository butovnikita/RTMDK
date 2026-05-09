"""Integration tests for SSE pipeline streaming endpoint."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField, RTMDKMemory

import rtmdk.server.app as app_mod


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


def _make_memory():
    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
    field = RTMDKField(cfg)
    field.add_node(
        embedding=np.array([0.0] * 16),
        content={"text": "hello world"},
        node_id="n0",
    )
    mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
    mem.field = field
    return mem


class TestPipelineSSEStream:
    def test_sse_content_type(self, client):
        mem = _make_memory()
        app_mod.memory = mem
        try:
            resp = client.get("/v1/memory/pipeline/stream?query=hello&top_k=5")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
        finally:
            app_mod.memory = None

    def test_sse_events_structure(self, client):
        mem = _make_memory()
        app_mod.memory = mem
        try:
            resp = client.get("/v1/memory/pipeline/stream?query=hello&top_k=5")
            body = resp.text
            lines = [l for l in body.split("\n") if l.startswith("data: ")]
            events = [json.loads(l[6:]) for l in lines]

            assert len(events) >= 2
            assert events[0]["event"] == "pipeline_started"
            assert events[-1]["event"] == "pipeline_completed"

            # Check pipeline_started has stage list
            assert "stages" in events[0]
            assert isinstance(events[0]["stages"], list)

            # Check pipeline_completed has results and metrics
            assert "results" in events[-1]
            assert "total_latency_ms" in events[-1]
            assert "metrics" in events[-1]
        finally:
            app_mod.memory = None

    def test_sse_no_memory_returns_error(self, client):
        resp = client.get("/v1/memory/pipeline/stream?query=hello&top_k=5")
        assert resp.status_code == 200
        # Error event embedded in SSE stream
        assert "error" in resp.text.lower()

    def test_sse_stage_events_present(self, client):
        mem = _make_memory()
        app_mod.memory = mem
        try:
            resp = client.get("/v1/memory/pipeline/stream?query=hello&top_k=5")
            body = resp.text
            lines = [l for l in body.split("\n") if l.startswith("data: ")]
            events = [json.loads(l[6:]) for l in lines]

            stage_events = [e for e in events if e["event"] == "stage_completed"]
            assert len(stage_events) >= 1
            for se in stage_events:
                assert "stage" in se
                assert "latency_ms" in se
                assert "breaker_state" in se
        finally:
            app_mod.memory = None

    def test_sse_query_sanitization(self, client):
        resp = client.get("/v1/memory/pipeline/stream?query=hello%00world&top_k=5")
        # Should be rejected before streaming starts
        assert resp.status_code == 400
