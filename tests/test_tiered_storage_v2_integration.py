"""Integration tests for Tiered Storage v2 (memmap-based) with RTMDKField."""
from __future__ import annotations

import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField


class TestTieredStorageV2Integration:
    def test_field_uses_tiered_v2_when_enabled(self, tmp_path):
        cfg = RTMDKConfig(
            latent_dim=64,
            max_nodes=100,
            tiered_storage_v2_enabled=True,
            tiered_storage_path=str(tmp_path / "cold"),
            tiered_hot_pct=0.1,
            tiered_warm_pct=0.3,
        )
        field = RTMDKField(config=cfg)
        from rtmdk.storage.tiered_adapter import TieredNodeStoreAdapter
        assert isinstance(field.nodes, TieredNodeStoreAdapter)

    def test_add_and_query_node_with_tiered_v2(self, tmp_path):
        cfg = RTMDKConfig(
            latent_dim=64,
            max_nodes=100,
            tiered_storage_v2_enabled=True,
            tiered_storage_path=str(tmp_path / "cold"),
            tiered_hot_pct=0.1,
            tiered_warm_pct=0.3,
            use_hnsw=False,
        )
        field = RTMDKField(config=cfg)
        emb = np.random.randn(64).astype(np.float32)
        node_id = field.add_node(emb, {"text": "hello"})
        assert node_id is not None
        # Query should find the node
        results = field.query(emb, top_k=5)
        assert len(results) > 0

    def test_tiered_v2_eviction_to_warm(self, tmp_path):
        cfg = RTMDKConfig(
            latent_dim=64,
            max_nodes=100,
            tiered_storage_v2_enabled=True,
            tiered_storage_path=str(tmp_path / "cold"),
            tiered_hot_pct=0.05,   # hot = 5 nodes
            tiered_warm_pct=0.10,  # warm = 10 nodes
            use_hnsw=False,
        )
        field = RTMDKField(config=cfg)
        for i in range(20):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"doc {i}"})
        # At least some nodes should be in warm/cold
        stats = field.nodes.stats()
        assert stats["hot_count"] <= 5
        assert stats["hot_count"] + stats["warm_count"] + stats["cold_count"] == 20

    def test_tiered_v2_stats(self, tmp_path):
        cfg = RTMDKConfig(
            latent_dim=64,
            max_nodes=100,
            tiered_storage_v2_enabled=True,
            tiered_storage_path=str(tmp_path / "cold"),
            use_hnsw=False,
        )
        field = RTMDKField(config=cfg)
        for i in range(5):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"doc {i}"})
        stats = field.nodes.stats()
        assert stats["total_puts"] == 5
        assert stats["hot_count"] > 0
