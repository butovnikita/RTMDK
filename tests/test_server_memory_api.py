"""Tests for RTMDK server memory query endpoints."""

import importlib

import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    # Disable API auth for tests
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def reset_memory():
    """Reset memory state between tests."""
    old = app_mod.memory
    app_mod.memory = None
    yield
    app_mod.memory = old


def test_memory_query_503_when_not_initialized(client):
    """Memory query returns 503 when memory not initialized."""
    resp = client.post("/v1/memory/query", json={"query": "hello", "top_k": 5})
    assert resp.status_code == 503
    assert "not initialized" in resp.json()["detail"]


def test_memory_batch_query_503_when_not_initialized(client):
    """Batch memory query returns 503 when memory not initialized."""
    resp = client.post("/v1/memory/batch_query", json={"queries": ["hello"], "top_k": 5})
    assert resp.status_code == 503
    assert "not initialized" in resp.json()["detail"]


def test_memory_query_validation(client):
    """Memory query validates input."""
    from rtmdk.memory.core import RTMDKField, RTMDKMemory

    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
    field = RTMDKField(cfg)
    mem = RTMDKMemory(config=cfg, embedder=lambda x: [0.0] * 16)
    mem.field = field
    app_mod.memory = mem

    try:
        # empty query
        resp = client.post("/v1/memory/query", json={"query": "", "top_k": 5})
        assert resp.status_code == 422

        # top_k too high
        resp = client.post("/v1/memory/query", json={"query": "x", "top_k": 100})
        assert resp.status_code == 422

        # negative threshold
        resp = client.post("/v1/memory/query", json={"query": "x", "threshold": -0.1})
        assert resp.status_code == 422
    finally:
        app_mod.memory = None


def test_memory_query_returns_results(client):
    """Memory query returns results when memory has nodes."""
    import numpy as np

    from rtmdk.memory.core import RTMDKField, RTMDKMemory

    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
    field = RTMDKField(cfg)
    field.add_node(embedding=np.array([0.0] * 16), content={"content": "hello world"}, node_id="n0")
    field.add_node(embedding=np.array([1.0] * 16), content={"content": "foo bar"}, node_id="n1")

    mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
    mem.field = field
    app_mod.memory = mem

    try:
        resp = client.post("/v1/memory/query", json={"query": "hello", "top_k": 2, "threshold": 0.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "hello"
        assert data["total"] >= 1
        assert len(data["results"]) >= 1
        assert "score" in data["results"][0]
        assert "content" in data["results"][0]
    finally:
        app_mod.memory = None


def test_memory_batch_query_returns_results(client):
    """Batch memory query returns results for all queries."""
    import numpy as np

    from rtmdk.memory.core import RTMDKField, RTMDKMemory

    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
    field = RTMDKField(cfg)
    field.add_node(embedding=np.array([0.0] * 16), content={"content": "hello world"}, node_id="n0")
    field.add_node(embedding=np.array([1.0] * 16), content={"content": "foo bar"}, node_id="n1")

    mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
    mem.field = field
    app_mod.memory = mem

    try:
        resp = client.post(
            "/v1/memory/batch_query",
            json={
                "queries": ["hello", "foo"],
                "top_k": 1,
                "threshold": 0.0,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["queries"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["query"] == "hello"
        assert data["latency_ms"] >= 0
    finally:
        app_mod.memory = None
