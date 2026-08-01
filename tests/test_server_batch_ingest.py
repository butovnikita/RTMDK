"""Tests for batch ingest endpoint."""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField, RTMDKMemory

app_mod = importlib.import_module("rtmdk.server.app")


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


def _make_mem():
    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
    field = RTMDKField(cfg)
    mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
    mem.field = field
    return mem


def test_batch_ingest_503_when_not_initialized(client):
    resp = client.post("/v1/memory/batch_ingest", json={"documents": ["a", "b"]})
    assert resp.status_code == 503


def test_batch_ingest(client):
    mem = _make_mem()
    app_mod.memory = mem
    try:
        resp = client.post(
            "/v1/memory/batch_ingest",
            json={
                "documents": ["doc1", "doc2", "doc3"],
                "metadata": {"source": "test"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ingested"] == 3
        assert len(data["node_ids"]) == 3
        assert data["latency_ms"] >= 0
        # Verify nodes exist
        for nid in data["node_ids"]:
            assert nid in mem.field.nodes
    finally:
        app_mod.memory = None


def test_batch_ingest_with_custom_ids(client):
    mem = _make_mem()
    app_mod.memory = mem
    try:
        resp = client.post(
            "/v1/memory/batch_ingest",
            json={
                "documents": ["doc1", "doc2"],
                "node_ids": ["custom_1", "custom_2"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_ids"] == ["custom_1", "custom_2"]
    finally:
        app_mod.memory = None


def test_batch_ingest_empty_rejected(client):
    resp = client.post("/v1/memory/batch_ingest", json={"documents": []})
    assert resp.status_code == 422
