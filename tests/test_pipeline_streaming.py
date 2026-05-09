"""Tests for pipeline SSE streaming."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField, RTMDKMemory
from rtmdk.pipeline.streaming import StreamingPipelineExecutor, _sse_event

app_mod = __import__("rtmdk.server.app", fromlist=["app"])


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


def make_memory():
    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
    field = RTMDKField(cfg)
    field.add_node(
        embedding=np.array([0.0] * 16),
        content={"content": "hello world"},
        node_id="n0",
    )
    mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
    mem.field = field
    return mem


class TestSSEEventFormatter:
    def test_sse_event_format(self):
        data = {"event": "test", "value": 42}
        out = _sse_event(data)
        assert out.startswith("data: ")
        assert out.endswith("\n\n")
        parsed = json.loads(out[6:-2])
        assert parsed == data


class TestStreamingPipelineExecutor:
    def test_stream_yields_events(self):
        mem = make_memory()
        pipeline = mem.build_pipeline()
        streamer = StreamingPipelineExecutor(pipeline.stages)
        chunks = list(streamer.run("hello", top_k=5))

        events = []
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            for line in lines:
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        assert events[0]["event"] == "pipeline_started"
        assert events[0]["query"] == "hello"
        assert "stages" in events[0]

        completed = [e for e in events if e["event"] == "stage_completed"]
        assert len(completed) >= 1

        assert events[-1]["event"] == "pipeline_completed"
        assert "results" in events[-1]
        assert "total_latency_ms" in events[-1]

    def test_stream_no_memory_returns_error(self):
        from rtmdk.pipeline.base import PipelineContext
        from rtmdk.pipeline.stages import EmbedStage

        class FakeEmbedder:
            def embed(self, text):
                raise RuntimeError("no memory")

        stage = EmbedStage(FakeEmbedder())
        streamer = StreamingPipelineExecutor([stage])
        chunks = list(streamer.run("hello", top_k=5))
        events = [json.loads(c[6:]) for c in chunks if c.startswith("data: ")]

        assert events[0]["event"] == "pipeline_started"
        assert events[-1]["event"] == "pipeline_completed"


class TestServerSSEEndpoint:
    def test_pipeline_stream_endpoint(self, client):
        mem = make_memory()
        app_mod.memory = mem

        try:
            resp = client.get("/v1/memory/pipeline/stream?query=hello&top_k=5")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

            body = resp.text
            lines = [l for l in body.split("\n") if l.startswith("data: ")]
            events = [json.loads(l[6:]) for l in lines]

            assert events[0]["event"] == "pipeline_started"
            assert events[-1]["event"] == "pipeline_completed"
            assert any(e["event"] == "stage_completed" for e in events)
        finally:
            app_mod.memory = None

    def test_pipeline_stream_no_memory(self, client):
        resp = client.get("/v1/memory/pipeline/stream?query=hello&top_k=5")
        assert resp.status_code == 200
        assert "error" in resp.text
