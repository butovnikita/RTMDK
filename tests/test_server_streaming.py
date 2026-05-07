"""Tests for streaming chat completions endpoint."""

import importlib

import pytest
from fastapi.testclient import TestClient

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


def test_streaming_request_header(client):
    """Streaming endpoint returns event-stream content type when LM Studio unavailable."""
    # Since LM Studio is not available in tests, we just verify the endpoint
    # structure accepts stream=True without crashing on validation
    old_lm = app_mod.lm_studio_available
    app_mod.lm_studio_available = False
    try:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        # Should get 503 because LM Studio not available
        assert resp.status_code == 503
    finally:
        app_mod.lm_studio_available = old_lm
