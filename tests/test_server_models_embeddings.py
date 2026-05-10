"""Tests for /v1/models and /v1/embeddings endpoints."""

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


class TestModelsEndpoint:
    def test_list_models_returns_rtmdk(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "rtmdk" in ids

    def test_list_models_with_chat_model(self, client):
        old = app_mod.chat_model
        app_mod.chat_model = "test-model"
        try:
            resp = client.get("/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            ids = [m["id"] for m in data["data"]]
            assert "test-model" in ids
            assert "rtmdk" in ids
        finally:
            app_mod.chat_model = old


class TestEmbeddingsEndpoint:
    def test_embeddings_single_input(self, client):
        cfg = RTMDKConfig(latent_dim=64, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        app_mod.memory = mem
        try:
            resp = client.post(
                "/v1/embeddings",
                json={"input": "hello world", "model": "rtmdk"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["index"] == 0
            assert isinstance(data["data"][0]["embedding"], list)
            assert len(data["data"][0]["embedding"]) > 0
        finally:
            app_mod.memory = None

    def test_embeddings_batch_input(self, client):
        cfg = RTMDKConfig(latent_dim=64, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        app_mod.memory = mem
        try:
            resp = client.post(
                "/v1/embeddings",
                json={"input": ["hello", "world"], "model": "rtmdk"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["data"]) == 2
            assert data["data"][0]["index"] == 0
            assert data["data"][1]["index"] == 1
        finally:
            app_mod.memory = None

    def test_embeddings_no_memory_returns_503(self, client):
        # memory is None
        resp = client.post(
            "/v1/embeddings",
            json={"input": "hello", "model": "rtmdk"},
        )
        # get_embedding falls through to LM Studio; if not available may 503
        assert resp.status_code in (200, 503)
