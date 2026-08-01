"""Tests for rtmdk.production.multi_tenant."""

import numpy as np
import pytest
from unittest.mock import patch
from rtmdk.production.multi_tenant import TenantRouter
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory


def _embed_factory():
    return lambda t: np.random.randn(768).astype(np.float32)


def _make_mem():
    cfg = RTMDKConfig(latent_dim=64)
    return RTMDKMemory(config=cfg, embedder=_embed_factory())


class TestTenantRouter:
    @patch("rtmdk.create_rtmdk", side_effect=lambda **kw: _make_mem())
    def test_get_tenant_memory_creates_new(self, mock_create):
        router = TenantRouter(_embed_factory, max_tenants=2)
        mem = router.get_tenant_memory("t1")
        assert mem is not None
        assert "t1" in router._memories

    @patch("rtmdk.create_rtmdk", side_effect=lambda **kw: _make_mem())
    def test_get_tenant_memory_returns_existing(self, mock_create):
        router = TenantRouter(_embed_factory, max_tenants=2)
        mem1 = router.get_tenant_memory("t1")
        mem2 = router.get_tenant_memory("t1")
        assert mem1 is mem2
        assert router._stats["t1"].query_count == 1

    @patch("rtmdk.create_rtmdk", side_effect=lambda **kw: _make_mem())
    def test_max_tenants(self, mock_create):
        router = TenantRouter(_embed_factory, max_tenants=1)
        router.get_tenant_memory("t1")
        with pytest.raises(ValueError):
            router.get_tenant_memory("t2")

    @patch("rtmdk.create_rtmdk", side_effect=lambda **kw: _make_mem())
    def test_remove_tenant(self, mock_create):
        router = TenantRouter(_embed_factory)
        router.get_tenant_memory("t1")
        assert router.remove_tenant("t1") is True
        assert router.remove_tenant("t1") is False

    @patch("rtmdk.create_rtmdk", side_effect=lambda **kw: _make_mem())
    def test_list_tenants(self, mock_create):
        router = TenantRouter(_embed_factory)
        router.get_tenant_memory("t1")
        tenants = router.list_tenants()
        assert len(tenants) == 1
        assert tenants[0]["tenant_id"] == "t1"

    @patch("rtmdk.create_rtmdk", side_effect=lambda **kw: _make_mem())
    def test_get_tenant_stats(self, mock_create):
        router = TenantRouter(_embed_factory)
        router.get_tenant_memory("t1")
        stats = router.get_tenant_stats("t1")
        assert stats is not None
        assert stats["tenant_id"] == "t1"

    def test_get_tenant_stats_missing(self):
        router = TenantRouter(_embed_factory)
        assert router.get_tenant_stats("missing") is None

    @patch("rtmdk.create_rtmdk", side_effect=lambda **kw: _make_mem())
    def test_get_global_stats(self, mock_create):
        router = TenantRouter(_embed_factory)
        router.get_tenant_memory("t1")
        gstats = router.get_global_stats()
        assert gstats["active_tenants"] == 1
        assert gstats["max_tenants"] > 0

    @patch("rtmdk.create_rtmdk", side_effect=lambda **kw: _make_mem())
    def test_callbacks(self, mock_create):
        created = []
        limited = []
        router = TenantRouter(
            _embed_factory,
            max_tenants=1,
            on_tenant_created=lambda t: created.append(t),
            on_tenant_limit=lambda t: limited.append(t),
        )
        router.get_tenant_memory("t1")
        assert created == ["t1"]
        with pytest.raises(ValueError):
            router.get_tenant_memory("t2")
        assert limited == ["t2"]
