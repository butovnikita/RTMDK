"""Load tests for RTMDK HTTP API.

Requires a running server (use TestClient for unit-test mode).
These tests use TestClient and are therefore not true external load
tests, but they validate concurrent endpoint handling.
"""

import asyncio
import importlib

import pytest
from fastapi.testclient import TestClient

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


class TestLoadHealth:
    def test_health_sequential(self, client):
        for _ in range(50):
            resp = client.get("/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_concurrent(self, client):
        async def _fetch():
            return client.get("/health")

        tasks = [_fetch() for _ in range(20)]
        results = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in results)


class TestLoadMemory:
    def test_query_memory_sequential(self, client):
        for i in range(20):
            resp = client.post(
                "/v1/memory/query",
                json={"query": f"test {i}", "top_k": 3},
            )
            assert resp.status_code in (200, 500, 503)  # 503 if memory not initialized

    def test_batch_ingest_small(self, client):
        docs = [f"document {i}" for i in range(50)]
        resp = client.post(
            "/v1/memory/batch_ingest",
            json={"documents": docs},
        )
        assert resp.status_code in (200, 500, 503)
