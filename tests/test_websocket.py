"""Tests for RTMDK WebSocket streaming endpoint."""

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


class TestWebSocketMemory:
    def test_websocket_ping_pong(self, client):
        with client.websocket_connect("/ws/memory") as ws:
            ws.send_json({"action": "ping"})
            resp = ws.receive_json()
            assert resp["type"] == "pong"

    def test_websocket_query_no_memory(self, client):
        with client.websocket_connect("/ws/memory") as ws:
            ws.send_json({"action": "query", "query": "hello", "top_k": 5})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "not ready" in resp["message"]

    def test_websocket_query_with_memory(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        field = RTMDKField(cfg)
        field.add_node(
            embedding=np.array([0.0] * 16),
            content={"content": "hello world"},
            node_id="n0")
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            with client.websocket_connect("/ws/memory") as ws:
                ws.send_json({"action": "query", "query": "hello", "top_k": 5})
                resp = ws.receive_json()
                assert resp["type"] == "query_results"
                assert "results" in resp
                assert len(resp["results"]) >= 1
                assert resp["results"][0]["node_id"] == "n0"
        finally:
            app_mod.memory = None

    def test_websocket_unknown_action(self, client):
        with client.websocket_connect("/ws/memory") as ws:
            ws.send_json({"action": "unknown"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Unknown action" in resp["message"]

    def test_websocket_invalid_json(self, client):
        with client.websocket_connect("/ws/memory") as ws:
            ws.send_text("not json")
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid JSON" in resp["message"]
