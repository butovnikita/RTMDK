"""Tests for APIKeyManager and TenantRateLimiter."""

import os
import tempfile

import pytest

from rtmdk.production.api_key_manager import APIKeyManager
from rtmdk.production.tenant_rate_limiter import TenantRateLimiter


@pytest.fixture
def key_mgr():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "keys.json")
        mgr = APIKeyManager(storage_path=path)
        yield mgr


class TestAPIKeyManager:
    def test_create_and_validate(self, key_mgr):
        raw, rec = key_mgr.create_key(tenant_id="t1", name="prod")
        assert raw.startswith("rtmdk_")
        assert rec.tenant_id == "t1"
        assert rec.name == "prod"
        assert key_mgr.validate_key(raw) == "t1"

    def test_validate_invalid_key(self, key_mgr):
        assert key_mgr.validate_key("invalid_key") is None

    def test_revoke_key(self, key_mgr):
        raw, rec = key_mgr.create_key(tenant_id="t1")
        assert key_mgr.validate_key(raw) == "t1"
        ok = key_mgr.revoke_key(rec.key_hash)
        assert ok is True
        assert key_mgr.validate_key(raw) is None

    def test_revoke_unknown_key(self, key_mgr):
        assert key_mgr.revoke_key("nope") is False

    def test_delete_key(self, key_mgr):
        raw, rec = key_mgr.create_key(tenant_id="t1")
        ok = key_mgr.delete_key(rec.key_hash)
        assert ok is True
        assert key_mgr.validate_key(raw) is None

    def test_list_keys(self, key_mgr):
        raw1, rec1 = key_mgr.create_key(tenant_id="t1", name="k1")
        raw2, rec2 = key_mgr.create_key(tenant_id="t2", name="k2")
        keys = key_mgr.list_keys()
        assert len(keys) == 2
        t1_keys = key_mgr.list_keys(tenant_id="t1")
        assert len(t1_keys) == 1
        assert t1_keys[0]["name"] == "k1"

    def test_rate_limit_override(self, key_mgr):
        raw, rec = key_mgr.create_key(
            tenant_id="t1",
            rate_limit_override={"per_minute": 200},
        )
        assert rec.rate_limit_override == {"per_minute": 200}

    def test_persistence(self, key_mgr):
        raw, rec = key_mgr.create_key(tenant_id="t1")
        # Re-instantiate manager pointing to same file
        mgr2 = APIKeyManager(storage_path=key_mgr.storage_path)
        assert mgr2.validate_key(raw) == "t1"


class TestTenantRateLimiter:
    def test_per_tenant_limits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "keys.json")
            mgr = APIKeyManager(storage_path=path)
            raw, rec = mgr.create_key(tenant_id="t1")
            trl = TenantRateLimiter(
                api_key_manager=mgr,
                default_per_minute=2,
                default_per_hour=10,
                default_per_day=100,
            )
            # First 2 requests allowed
            assert trl.allow_request("t1") is True
            assert trl.allow_request("t1") is True
            # Third blocked
            assert trl.allow_request("t1") is False

    def test_tenant_isolation(self):
        trl = TenantRateLimiter(
            default_per_minute=1,
            default_per_hour=10,
            default_per_day=100,
        )
        assert trl.allow_request("t1") is True
        assert trl.allow_request("t1") is False
        assert trl.allow_request("t2") is True

    def test_remaining(self):
        trl = TenantRateLimiter(
            default_per_minute=5,
            default_per_hour=10,
            default_per_day=100,
        )
        trl.allow_request("t1")
        rem = trl.get_remaining("t1")
        assert rem["per_minute"] == 4
        assert rem["per_hour"] == 9
        assert rem["per_day"] == 99
