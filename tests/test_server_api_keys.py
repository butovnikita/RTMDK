"""Tests for RTMDK server API key management endpoints."""

import importlib

import pytest
from fastapi.testclient import TestClient

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    # Disable API auth for most tests, but keep manager available
    app_mod.ENABLE_API_AUTH = False
    # Ensure api_key_manager exists
    if app_mod.api_key_manager is None:
        from rtmdk.production.api_key_manager import APIKeyManager
        import tempfile
        app_mod.api_key_manager = APIKeyManager(
            storage_path=tempfile.mktemp(suffix=".json")
        )
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def reset_keys():
    old_mgr = app_mod.api_key_manager
    import tempfile
    from rtmdk.production.api_key_manager import APIKeyManager
    app_mod.api_key_manager = APIKeyManager(
        storage_path=tempfile.mktemp(suffix=".json")
    )
    yield
    app_mod.api_key_manager = old_mgr


class TestAdminAPIKeys:
    def test_create_key_requires_admin(self, client):
        """Without admin key, creation is forbidden when auth enabled."""
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            # No admin key in request state because auth middleware skipped (auth disabled fixture)
            # Actually with ENABLE_API_AUTH=True and no valid key, security middleware blocks
            resp = client.post(
                "/v1/admin/api-keys",
                json={"tenant_id": "t1", "name": "test"},
                headers={"Authorization": "Bearer invalid"},
            )
            assert resp.status_code == 401
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_create_key_with_admin(self, client):
        """Admin can create keys."""
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            resp = client.post(
                "/v1/admin/api-keys",
                json={"tenant_id": "t1", "name": "test-key"},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["api_key"].startswith("rtmdk_")
            assert data["tenant_id"] == "t1"
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_list_keys(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            # Create a key first
            create_resp = client.post(
                "/v1/admin/api-keys",
                json={"tenant_id": "t1", "name": "k1"},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert create_resp.status_code == 200

            resp = client.get(
                "/v1/admin/api-keys",
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["keys"]) >= 1
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_revoke_key(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            create_resp = client.post(
                "/v1/admin/api-keys",
                json={"tenant_id": "t1", "name": "k2"},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            key_hash = create_resp.json()["key_hash"]

            resp = client.post(
                "/v1/admin/api-keys/revoke",
                json={"key_hash": key_hash},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "revoked"

            # Verify key no longer works
            raw_key = create_resp.json()["api_key"]
            assert app_mod.api_key_manager.validate_key(raw_key) is None
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_revoke_unknown_key(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            resp = client.post(
                "/v1/admin/api-keys/revoke",
                json={"key_hash": "nonexistent"},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert resp.status_code == 404
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_delete_key(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            create_resp = client.post(
                "/v1/admin/api-keys",
                json={"tenant_id": "t1", "name": "k3"},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            key_hash = create_resp.json()["key_hash"]

            resp = client.delete(
                f"/v1/admin/api-keys/{key_hash}",
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_list_tenants(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            client.post(
                "/v1/admin/api-keys",
                json={"tenant_id": "t1", "name": "k1"},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            client.post(
                "/v1/admin/api-keys",
                json={"tenant_id": "t2", "name": "k2"},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            resp = client.get(
                "/v1/admin/tenants",
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["tenants"]) == 2
        finally:
            app_mod.ENABLE_API_AUTH = old_auth


class TestTenantAuthFlow:
    def test_tenant_key_blocks_invalid(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            resp = client.post(
                "/v1/memory/query",
                json={"query": "hello"},
                headers={"X-API-Key": "invalid_key"},
            )
            assert resp.status_code == 401
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_tenant_key_allows_valid(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            # Create a tenant key
            create_resp = client.post(
                "/v1/admin/api-keys",
                json={"tenant_id": "t1"},
                headers={"X-API-Key": app_mod.API_KEY},
            )
            tenant_key = create_resp.json()["api_key"]

            # Valid tenant key should pass auth (memory may be uninitialized -> 503)
            resp = client.post(
                "/v1/memory/query",
                json={"query": "hello"},
                headers={"X-API-Key": tenant_key},
            )
            # 503 means auth passed but memory not initialized in test
            assert resp.status_code == 503
        finally:
            app_mod.ENABLE_API_AUTH = old_auth
