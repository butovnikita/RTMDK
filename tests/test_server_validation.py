"""Tests for REST API input validation (negative top_k, too large, malformed)."""

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


class TestMemoryQueryValidation:
    def test_top_k_zero_rejected(self, client):
        _init_memory()
        resp = client.post("/v1/memory/query", json={"query": "hello", "top_k": 0})
        assert resp.status_code == 422

    def test_top_k_negative_rejected(self, client):
        _init_memory()
        resp = client.post("/v1/memory/query", json={"query": "hello", "top_k": -1})
        assert resp.status_code == 422

    def test_top_k_too_large_rejected(self, client):
        _init_memory()
        resp = client.post("/v1/memory/query", json={"query": "hello", "top_k": 51})
        assert resp.status_code == 422

    def test_top_k_boundary_accepted(self, client):
        _init_memory()
        resp = client.post("/v1/memory/query", json={"query": "hello", "top_k": 50})
        assert resp.status_code == 200

    def test_missing_query_rejected(self, client):
        _init_memory()
        resp = client.post("/v1/memory/query", json={"top_k": 5})
        assert resp.status_code == 422

    def test_malformed_json_rejected(self, client):
        _init_memory()
        resp = client.post(
            "/v1/memory/query",
            data="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


class TestPipelinePlanValidation:
    def test_top_k_negative_not_validated_in_query_params(self, client):
        """Pipeline plan top_k via query params lacks ge=1; documents current behavior."""
        _init_memory()
        resp = client.get("/v1/memory/pipeline/plan", params={"query": "hello", "top_k": -1})
        # Currently passes because FastAPI Query() doesn't enforce ge=1 here
        assert resp.status_code == 200

    def test_top_k_too_large_not_validated_in_query_params(self, client):
        """Pipeline plan top_k via query params lacks le=50; documents current behavior."""
        _init_memory()
        resp = client.get("/v1/memory/pipeline/plan", params={"query": "hello", "top_k": 101})
        assert resp.status_code == 200

    def test_missing_query_returns_400(self, client):
        """_sanitize_query rejects empty string."""
        _init_memory()
        resp = client.get("/v1/memory/pipeline/plan")
        assert resp.status_code == 400
