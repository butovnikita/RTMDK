"""Tests for tiered node storage prototype."""
from __future__ import annotations

import numpy as np
import pytest

from rtmdk.storage.tiered import TieredNodeStore


class TestTieredNodeStore:
    def test_put_and_get_hot(self, tmp_path):
        store = TieredNodeStore(max_hot=10, max_warm=100, cold_dir=str(tmp_path / "cold"))
        data = {"text": "hello", "embedding": np.random.randn(384).astype(np.float32)}
        store.put("node_1", data)
        retrieved = store.get("node_1")
        assert retrieved is not None
        assert retrieved["text"] == "hello"
        assert np.allclose(retrieved["embedding"], data["embedding"])
        stats = store.stats()
        assert stats["hot_count"] == 1
        store.close()

    def test_get_missing_returns_none(self, tmp_path):
        store = TieredNodeStore(max_hot=10, max_warm=100, cold_dir=str(tmp_path / "cold"))
        assert store.get("missing") is None
        store.close()

    def test_hot_eviction_to_warm(self, tmp_path):
        store = TieredNodeStore(max_hot=2, max_warm=10, cold_dir=str(tmp_path / "cold"))
        for i in range(3):
            store.put(f"node_{i}", {
                "text": f"text_{i}",
                "embedding": np.random.randn(384).astype(np.float32),
            })
        stats = store.stats()
        assert stats["hot_count"] == 2  # max_hot=2
        assert stats["warm_count"] == 1
        store.close()

    def test_warm_promotion_on_access(self, tmp_path):
        store = TieredNodeStore(max_hot=2, max_warm=10, cold_dir=str(tmp_path / "cold"))
        store.put("node_a", {"text": "a", "embedding": np.random.randn(384).astype(np.float32)})
        store.put("node_b", {"text": "b", "embedding": np.random.randn(384).astype(np.float32)})
        store.put("node_c", {"text": "c", "embedding": np.random.randn(384).astype(np.float32)})
        # node_a should be in warm
        stats_before = store.stats()
        assert stats_before["warm_count"] == 1
        # Access node_a -> promote to hot
        retrieved = store.get("node_a")
        assert retrieved is not None
        stats_after = store.stats()
        assert stats_after["promotions"] >= 1
        store.close()

    def test_delete_removes_from_all_tiers(self, tmp_path):
        store = TieredNodeStore(max_hot=2, max_warm=10, cold_dir=str(tmp_path / "cold"))
        store.put("node_x", {"text": "x", "embedding": np.random.randn(384).astype(np.float32)})
        store.delete("node_x")
        assert store.get("node_x") is None
        store.close()

    def test_stats_traffic(self, tmp_path):
        store = TieredNodeStore(max_hot=10, max_warm=100, cold_dir=str(tmp_path / "cold"))
        store.put("n1", {"text": "1", "embedding": np.random.randn(384).astype(np.float32)})
        store.get("n1")
        store.get("n1")
        stats = store.stats()
        assert stats["total_puts"] == 1
        assert stats["total_gets"] == 2
        store.close()

    def test_context_manager(self, tmp_path):
        with TieredNodeStore(max_hot=10, max_warm=100, cold_dir=str(tmp_path / "cold")) as store:
            store.put("n1", {"text": "1", "embedding": np.random.randn(384).astype(np.float32)})
            assert store.get("n1") is not None
        # After exit, should be safe to re-open

    def test_cold_tier_persistence(self, tmp_path):
        cold_dir = tmp_path / "cold"
        store = TieredNodeStore(max_hot=1, max_warm=1, cold_dir=str(cold_dir))
        for i in range(3):
            store.put(f"node_{i}", {
                "text": f"text_{i}",
                "embedding": np.random.randn(384).astype(np.float32),
            })
        stats = store.stats()
        assert stats["cold_count"] >= 1
        # Re-open and verify cold data still accessible
        store.close()
        store2 = TieredNodeStore(max_hot=1, max_warm=1, cold_dir=str(cold_dir))
        retrieved = store2.get("node_0")
        assert retrieved is not None or store2.get("node_1") is not None or store2.get("node_2") is not None
        store2.close()
