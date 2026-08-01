"""Tests for RTMDK server memory node CRUD endpoints."""

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


def test_create_node_503_when_not_initialized(client):
    resp = client.post("/v1/memory/nodes", json={"content": "hello"})
    assert resp.status_code == 503


def test_create_node(client):
    mem = _make_mem()
    app_mod.memory = mem
    try:
        resp = client.post(
            "/v1/memory/nodes",
            json={
                "content": "hello world",
                "node_id": "n_test_1",
                "metadata": {"tag": "greeting"},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "n_test_1"
        assert data["status"] == "created"
    finally:
        app_mod.memory = None


def test_get_node(client):
    mem = _make_mem()
    mem.field.add_node(
        embedding=np.array([0.0] * 16),
        content={"content": "test"},
        node_id="n1",
    )
    app_mod.memory = mem
    try:
        resp = client.get("/v1/memory/nodes/n1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "n1"
        assert data["content"]["content"] == "test"
    finally:
        app_mod.memory = None


def test_get_node_not_found(client):
    mem = _make_mem()
    app_mod.memory = mem
    try:
        resp = client.get("/v1/memory/nodes/nonexistent")
        assert resp.status_code == 404
    finally:
        app_mod.memory = None


def test_update_node(client):
    mem = _make_mem()
    mem.field.add_node(
        embedding=np.array([0.0] * 16),
        content={"content": "old"},
        node_id="n1",
    )
    app_mod.memory = mem
    try:
        resp = client.put(
            "/v1/memory/nodes/n1",
            json={
                "content": "new",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"
    finally:
        app_mod.memory = None


def test_delete_node(client):
    mem = _make_mem()
    mem.field.add_node(
        embedding=np.array([0.0] * 16),
        content={"content": "to delete"},
        node_id="n1",
    )
    app_mod.memory = mem
    try:
        resp = client.delete("/v1/memory/nodes/n1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"
        assert "n1" not in mem.field.nodes
    finally:
        app_mod.memory = None


def test_list_nodes(client):
    mem = _make_mem()
    mem.field.add_node(
        embedding=np.array([0.0] * 16),
        content={"content": "a"},
        node_id="n1",
    )
    mem.field.add_node(
        embedding=np.array([0.1] * 16),
        content={"content": "b"},
        node_id="n2",
    )
    app_mod.memory = mem
    try:
        resp = client.get("/v1/memory/nodes?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["nodes"]) == 2
        assert data["offset"] == 0
        assert data["limit"] == 10
    finally:
        app_mod.memory = None


def test_list_nodes_pagination(client):
    mem = _make_mem()
    for i in range(5):
        mem.field.add_node(
            embedding=np.array([0.0] * 16),
            content={"content": f"item{i}"},
            node_id=f"n{i}",
        )
    app_mod.memory = mem
    try:
        resp = client.get("/v1/memory/nodes?limit=2&offset=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["nodes"]) == 2
    finally:
        app_mod.memory = None


def test_get_node_format_matches_list_nodes(client):
    """Regression: get_node and list_nodes must return content in the same
    dict format (keys and structure must match)."""
    mem = _make_mem()
    mem.field.add_node(
        embedding=np.array([0.0] * 16),
        content={"content": "hello", "tag": "greeting"},
        node_id="n_fmt",
    )
    app_mod.memory = mem
    try:
        get_resp = client.get("/v1/memory/nodes/n_fmt")
        assert get_resp.status_code == 200
        get_data = get_resp.json()

        list_resp = client.get("/v1/memory/nodes?limit=1")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        list_node = list_data["nodes"][0]

        assert get_data["content"] == list_node["content"]
        assert "salience" in get_data
        assert "salience" in list_node
    finally:
        app_mod.memory = None


def test_list_nodes_uses_lazy_islice(client):
    """Regression: list_nodes should not materialize the full node list
    into memory before slicing."""
    mem = _make_mem()
    for i in range(100):
        mem.field.add_node(
            embedding=np.array([0.0] * 16),
            content={"content": f"node{i}"},
            node_id=f"n{i}",
        )
    app_mod.memory = mem
    try:
        resp = client.get("/v1/memory/nodes?limit=5&offset=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 100
        assert len(data["nodes"]) == 5
        assert data["offset"] == 10
    finally:
        app_mod.memory = None
