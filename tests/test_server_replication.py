"""Tests for replication HTTP endpoints (/v1/replication/*)."""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory
from rtmdk.production.replication import ReplicationManager

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


def _init_memory_with_replication(tmp_path):
    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, wal_fsync_interval_ms=0)
    mem = RTMDKMemory(config=cfg, embedder=_embedder)
    # ReplicationManager needs peers to be enabled
    rm = ReplicationManager(peers=["http://peer:8000"], wal_path=str(tmp_path / "rep.db"))
    object.__setattr__(mem, "replication_manager", rm)
    app_mod.memory = mem
    return mem


class TestReplicationMutation:
    def test_receive_mutation_success(self, client, tmp_path):
        _init_memory_with_replication(tmp_path)
        resp = client.post(
            "/v1/replication/mutation",
            json={
                "node_id": "n1",
                "action": "add",
                "content": {"text": "hello"},
                "_rep_clock": 42,
                "_rep_origin": "node_b",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["clock"] == 42

    def test_receive_mutation_no_memory(self, client):
        resp = client.post(
            "/v1/replication/mutation",
            json={"node_id": "n1", "action": "add"},
        )
        assert resp.status_code == 503

    def test_receive_mutation_not_enabled(self, client, tmp_path):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        # ReplicationManager without peers -> disabled
        rm = ReplicationManager(peers=[], wal_path=str(tmp_path / "rep.db"))
        object.__setattr__(mem, "replication_manager", rm)
        app_mod.memory = mem
        try:
            resp = client.post(
                "/v1/replication/mutation",
                json={"node_id": "n1", "action": "add"},
            )
            assert resp.status_code == 503
            assert "not enabled" in resp.json()["detail"]
        finally:
            app_mod.memory = None


class TestReplicationWAL:
    def test_get_wal_empty(self, client, tmp_path):
        _init_memory_with_replication(tmp_path)
        resp = client.get("/v1/replication/wal")
        assert resp.status_code == 200
        data = resp.json()
        assert data["mutations"] == []
        assert "node_id" in data

    def test_get_wal_with_entries(self, client, tmp_path):
        _init_memory_with_replication(tmp_path)
        # Append a mutation first
        client.post(
            "/v1/replication/mutation",
            json={
                "node_id": "n1",
                "action": "add",
                "_rep_clock": 1,
                "_rep_origin": "peer",
            },
        )
        resp = client.get("/v1/replication/wal")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["mutations"]) == 1

    def test_get_wal_since_filter(self, client, tmp_path):
        mem = _init_memory_with_replication(tmp_path)
        rm = mem.replication_manager
        rm._wal.append(1, "peer", {"action": "add"})
        rm._wal.append(5, "peer", {"action": "del"})
        resp = client.get("/v1/replication/wal?since=2")
        assert resp.status_code == 200
        data = resp.json()
        clocks = [m["_rep_clock"] for m in data["mutations"]]
        assert clocks == [5]

    def test_get_wal_no_memory(self, client):
        resp = client.get("/v1/replication/wal")
        assert resp.status_code == 503
