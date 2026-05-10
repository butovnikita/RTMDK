"""Tests for pipeline visualization endpoints and models list."""

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


class TestModelsEndpoint:
    def test_list_models_returns_rtmdk(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert any(m["id"] == "rtmdk" for m in data["data"])


class TestPipelineDag:
    def test_pipeline_dag_503_when_not_initialized(self, client):
        resp = client.get("/v1/memory/pipeline/dag")
        assert resp.status_code == 503
        assert "not initialized" in resp.json()["detail"]

    def test_pipeline_dag_returns_stages(self, client):
        _init_memory()
        resp = client.get("/v1/memory/pipeline/dag")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert data["total_stages"] > 0
        assert data["enabled_stages"] > 0


class TestPipelinePlan:
    def test_pipeline_plan_503_when_not_initialized(self, client):
        resp = client.get("/v1/memory/pipeline/plan", params={"query": "test"})
        assert resp.status_code == 503
        assert "not initialized" in resp.json()["detail"]

    def test_pipeline_plan_returns_plan(self, client):
        _init_memory()
        resp = client.get("/v1/memory/pipeline/plan", params={"query": "hello", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "hello"
        assert "plan" in data


class TestPipelineHealth:
    def test_pipeline_health_503_when_not_initialized(self, client):
        resp = client.get("/v1/memory/pipeline/health")
        assert resp.status_code == 503
        assert "not initialized" in resp.json()["detail"]

    def test_pipeline_health_returns_stages(self, client):
        _init_memory()
        resp = client.get("/v1/memory/pipeline/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "stages" in data
        assert len(data["stages"]) > 0
