"""Tests for /v1/admin/config hot-reload endpoint."""

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


class TestAdminConfigReload:
    def test_config_reload_no_memory(self, client):
        resp = client.post("/v1/admin/config", json={"top_k": 10})
        assert resp.status_code == 503

    def test_config_reload_valid_field(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        app_mod.memory = mem
        try:
            old_top_k = mem.config.top_k
            resp = client.post("/v1/admin/config", json={"top_k": 20})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "updated"
            assert "top_k" in data["fields"]
            assert mem.config.top_k == 20
        finally:
            mem.config.top_k = old_top_k
            app_mod.memory = None

    def test_config_reload_multiple_fields(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        app_mod.memory = mem
        try:
            resp = client.post(
                "/v1/admin/config",
                json={"top_k": 15, "decay_rate": 0.999},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert set(data["fields"]) == {"top_k", "decay_rate"}
        finally:
            app_mod.memory = None

    def test_config_reload_invalid_field_rejected(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        app_mod.memory = mem
        try:
            resp = client.post("/v1/admin/config", json={"latent_dim": 32})
            assert resp.status_code == 400
            assert "not allowed" in resp.json()["detail"]
        finally:
            app_mod.memory = None

    def test_config_reload_requires_admin(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            resp = client.post(
                "/v1/admin/config",
                json={"top_k": 10},
                headers={"Authorization": "Bearer invalid"},
            )
            assert resp.status_code == 401
        finally:
            app_mod.ENABLE_API_AUTH = old_auth
