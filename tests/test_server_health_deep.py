"""Tests for /health/deep endpoint."""

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


class TestHealthDeep:
    def test_deep_health_no_memory(self, client):
        resp = client.get("/health/deep")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "memory_field" in data["checks"]
        assert data["checks"]["memory_field"]["status"] == "error"

    def test_deep_health_with_memory(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        for i in range(5):
            mem.add_node(content={"text": f"node {i}"}, embedding=np.array([0.0] * 16))
        app_mod.memory = mem
        try:
            resp = client.get("/health/deep")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            checks = data["checks"]
            assert checks["memory_field"]["status"] == "ok"
            assert checks["memory_field"]["nodes"] == 5
            assert checks["embedding_dims"]["status"] == "ok"
            assert checks["embedding_dims"]["expected"] == 16
            assert "wal" in checks
            assert "async_index" in checks
            assert "active_requests" in checks
        finally:
            app_mod.memory = None

    def test_deep_health_embedding_dim_mismatch(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        mem.add_node(content={"text": "good"}, embedding=np.array([0.0] * 16))
        # Manually corrupt a node to trigger dim mismatch
        bad_node = list(mem.field.nodes.values())[0]
        bad_node.latent_pos = np.array([0.0] * 8)
        app_mod.memory = mem
        try:
            resp = client.get("/health/deep")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "degraded"
            assert data["checks"]["embedding_dims"]["status"] == "error"
        finally:
            app_mod.memory = None
