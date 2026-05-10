"""Tests for RTMDK server admin endpoints (audit log, retention)."""

import importlib

import pytest
from fastapi.testclient import TestClient

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    app_mod.ENABLE_API_AUTH = False
    if app_mod.audit_log is None:
        from rtmdk.production.audit_log import AuditLog
        import tempfile

        app_mod.audit_log = AuditLog(storage_path=tempfile.mktemp(suffix=".json"))
    if app_mod.retention_manager is None:
        from rtmdk.production.retention import RetentionManager

        app_mod.retention_manager = RetentionManager(None)
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def reset_state():
    old_log = app_mod.audit_log
    old_ret = app_mod.retention_manager
    from rtmdk.production.audit_log import AuditLog
    from rtmdk.production.retention import RetentionManager
    import tempfile

    app_mod.audit_log = AuditLog(storage_path=tempfile.mktemp(suffix=".json"))
    app_mod.retention_manager = RetentionManager(None)
    yield
    app_mod.audit_log = old_log
    app_mod.retention_manager = old_ret


class TestAuditLog:
    def test_audit_log_query_requires_admin(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            resp = client.get(
                "/v1/admin/audit-log",
                headers={"Authorization": "Bearer invalid"},
            )
            assert resp.status_code == 401
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_audit_log_query_with_admin(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            resp = client.get(
                "/v1/admin/audit-log",
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "entries" in data
            assert "count" in data
        finally:
            app_mod.ENABLE_API_AUTH = old_auth


class TestRetention:
    def test_retention_stats_requires_admin(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            resp = client.get(
                "/v1/admin/retention",
                headers={"Authorization": "Bearer invalid"},
            )
            assert resp.status_code == 401
        finally:
            app_mod.ENABLE_API_AUTH = old_auth

    def test_retention_stats_with_admin(self, client):
        old_auth = app_mod.ENABLE_API_AUTH
        app_mod.ENABLE_API_AUTH = True
        try:
            resp = client.get(
                "/v1/admin/retention",
                headers={"X-API-Key": app_mod.API_KEY},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "pruned_total" in data
            assert "policy" in data
        finally:
            app_mod.ENABLE_API_AUTH = old_auth


class TestCacheStats:
    def test_cache_stats_no_auth(self, client):
        # ENABLE_API_AUTH is False in fixture
        resp = client.get("/v1/admin/cache")
        assert resp.status_code == 200
        data = resp.json()
        assert "redis_query" in data
        assert "redis_embedding" in data


class TestEncryptionStatus:
    def test_encryption_not_available(self, client):
        old = app_mod.encryption_manager
        app_mod.encryption_manager = None
        try:
            resp = client.get("/v1/admin/encryption")
            assert resp.status_code == 503
        finally:
            app_mod.encryption_manager = old


class TestTelemetryStatus:
    def test_telemetry_not_available(self, client):
        old = app_mod.telemetry_manager
        app_mod.telemetry_manager = None
        try:
            resp = client.get("/v1/admin/telemetry")
            assert resp.status_code == 503
        finally:
            app_mod.telemetry_manager = old
