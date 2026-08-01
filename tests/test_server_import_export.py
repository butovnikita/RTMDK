"""Tests for memory import/export endpoints."""

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


def test_export_503_when_not_initialized(client):
    resp = client.get("/v1/memory/export")
    assert resp.status_code == 503


def test_export_returns_nodes(client):
    mem = _make_mem()
    mem.field.add_node(
        embedding=np.array([0.0] * 16),
        content={"content": "hello"},
        node_id="n1",
    )
    app_mod.memory = mem
    try:
        resp = client.get("/v1/memory/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["id"] == "n1"
    finally:
        app_mod.memory = None


def test_import_nodes(client):
    mem = _make_mem()
    app_mod.memory = mem
    try:
        resp = client.post(
            "/v1/memory/import",
            json={
                "nodes": [
                    {
                        "id": "imp1",
                        "embedding": [0.0] * 16,
                        "content": {"content": "imported"},
                    }
                ],
                "clear_existing": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["imported"] == 1
        assert "imp1" in mem.field.nodes
    finally:
        app_mod.memory = None


def test_import_with_clear(client):
    mem = _make_mem()
    mem.field.add_node(
        embedding=np.array([0.1] * 16),
        content={"content": "old"},
        node_id="old1",
    )
    app_mod.memory = mem
    try:
        resp = client.post(
            "/v1/memory/import",
            json={
                "nodes": [
                    {
                        "id": "new1",
                        "embedding": [0.0] * 16,
                        "content": {"content": "new"},
                    }
                ],
                "clear_existing": True,
            },
        )
        assert resp.status_code == 200
        assert "old1" not in mem.field.nodes
        assert "new1" in mem.field.nodes
    finally:
        app_mod.memory = None


def test_import_empty_rejected(client):
    resp = client.post("/v1/memory/import", json={"nodes": []})
    assert resp.status_code == 422
