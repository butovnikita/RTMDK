"""Tests for structured request logging middleware."""

import importlib

import pytest
from fastapi.testclient import TestClient

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


def test_request_id_header_present(client):
    """Every response includes X-Request-ID header."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    assert len(resp.headers["x-request-id"]) == 8


def test_request_id_unique_per_request(client):
    """Each request gets a unique request ID."""
    resp1 = client.get("/health")
    resp2 = client.get("/health")
    assert resp1.headers["x-request-id"] != resp2.headers["x-request-id"]
