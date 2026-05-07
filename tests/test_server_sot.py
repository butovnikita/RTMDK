"""Tests for SOT REST endpoints."""

import importlib

import pytest
from fastapi.testclient import TestClient

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


class TestSOTStatus:
    def test_sot_status(self, client):
        resp = client.get("/v1/sot/status")
        # Memory may or may not be initialized depending on test order
        assert resp.status_code in (200, 503)

    def test_sot_bootstrap_no_memory(self, client):
        resp = client.post("/v1/sot/bootstrap", json={"texts": ["hello"]})
        # If memory not initialized → 503
        assert resp.status_code in (200, 503)


class TestSOTVocab:
    def test_vocab_when_no_memory(self, client):
        resp = client.get("/v1/sot/vocab")
        assert resp.status_code == 503

    def test_vocab_with_sot_enabled(self, client):
        import numpy as np
        from rtmdk.memory.config import RTMDKConfig
        from rtmdk.memory.core import RTMDKField, RTMDKMemory

        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, sot_enabled=True)
        field = RTMDKField(cfg)
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.get("/v1/sot/vocab?limit=10&offset=0")
            assert resp.status_code == 200
            data = resp.json()
            assert "items" in data
            assert "total" in data
            assert data["limit"] == 10
            assert data["offset"] == 0
        finally:
            app_mod.memory = None

    def test_vocab_search_filter(self, client):
        import numpy as np
        from rtmdk.memory.config import RTMDKConfig
        from rtmdk.memory.core import RTMDKField, RTMDKMemory

        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, sot_enabled=True)
        field = RTMDKField(cfg)
        # Seed a word-level mapping if available
        if field.sot_tokenizer:
            field.sot_tokenizer.word_to_id["hello"] = 256
            field.sot_tokenizer.id_to_word[256] = "hello"
            field.sot_tokenizer.token_embeddings[256] = np.zeros(16, dtype=np.float32)
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.get("/v1/sot/vocab?search=hello")
            assert resp.status_code == 200
            data = resp.json()
            # Should include hello or be empty if filter doesn't match
            assert "items" in data
        finally:
            app_mod.memory = None
