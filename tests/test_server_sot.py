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
