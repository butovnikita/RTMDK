"""Security tests for pipeline endpoints."""

import pytest
from fastapi.testclient import TestClient

import rtmdk.server.app as app_mod


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


class TestQuerySanitization:
    def test_null_bytes_rejected(self, client):
        resp = client.post("/v1/memory/query_pipeline", json={
            "query": "hello\x00world",
            "top_k": 5,
        })
        assert resp.status_code == 400
        assert "invalid" in resp.json()["detail"].lower()

    def test_control_chars_stripped(self, client):
        # Even without memory, sanitization should run first
        resp = client.post("/v1/memory/query_pipeline", json={
            "query": "hello\x01\x02world",
            "top_k": 5,
        })
        # Should pass sanitization but fail on memory not initialized
        assert resp.status_code == 503  # memory not ready, not 400

    def test_whitespace_normalized(self, client):
        resp = client.post("/v1/memory/query_pipeline", json={
            "query": "hello    world\t\t\ntest",
            "top_k": 5,
        })
        assert resp.status_code == 503  # sanitized, but memory not ready

    def test_max_length_enforced(self, client):
        long_query = "a" * 5000
        resp = client.post("/v1/memory/query_pipeline", json={
            "query": long_query,
            "top_k": 5,
        })
        assert resp.status_code == 400
        assert "max length" in resp.json()["detail"].lower()

    def test_empty_query_rejected(self, client):
        resp = client.post("/v1/memory/query_pipeline", json={
            "query": "",
            "top_k": 5,
        })
        # Pydantic validates min_length=1 before our sanitize
        assert resp.status_code == 422

    def test_stream_sanitization(self, client):
        resp = client.get("/v1/memory/pipeline/stream?query=hello%00world&top_k=5")
        assert resp.status_code == 400

    def test_valid_query_passes(self, client):
        resp = client.post("/v1/memory/query_pipeline", json={
            "query": "What is resonance?",
            "top_k": 5,
        })
        # Should pass sanitization, fail on memory
        assert resp.status_code == 503
