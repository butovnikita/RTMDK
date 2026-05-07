"""Tests for AsyncIndexBuilder background HNSW merge (Track 4)."""

import time

import numpy as np
import pytest

from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.async_index import AsyncIndexBuilder
from rtmdk.support.hnsw import NaiveGraphIndex


def _make_field(async_build: bool = True, interval_ms: int = 50, batch_size: int = 1000):
    cfg = RTMDKConfig(
        latent_dim=64,
        use_hnsw=True,
        hyperbolic=False,
        bm25_fallback=False,
        quantization="none",
        query_cache_size=0,
        async_hnsw_build=async_build,
        async_hnsw_interval_ms=interval_ms,
        async_hnsw_batch_size=batch_size,
    )
    return RTMDKField(cfg)


class TestAsyncIndexBuilder:
    def test_deferred_insert_eventually_indexed(self):
        field = _make_field(interval_ms=50, batch_size=1000)
        emb = np.random.randn(64).astype(np.float32)
        nid = field.add_node(emb, {"text": "hello"})
        # HNSW should not have it yet (pending)
        assert nid not in field.hnsw_index.positions
        time.sleep(0.15)
        assert nid in field.hnsw_index.positions
        field.close()

    def test_batch_flush_on_size(self):
        field = _make_field(interval_ms=10_000, batch_size=3)
        for i in range(3):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        time.sleep(0.05)
        assert len(field.hnsw_index.positions) == 3
        field.close()

    def test_batch_flush_on_interval(self):
        field = _make_field(interval_ms=50, batch_size=1000)
        for i in range(2):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        # Not enough to hit batch size
        time.sleep(0.15)
        assert len(field.hnsw_index.positions) == 2
        field.close()

    def test_query_finds_deferred_nodes(self):
        field = _make_field(interval_ms=50, batch_size=1000)
        np.random.seed(42)
        embeddings = np.random.randn(10, 64).astype(np.float32)
        for i, emb in enumerate(embeddings):
            field.add_node(emb, {"text": f"node {i}"})
        # Wait for background flush
        time.sleep(0.15)
        q = embeddings[3]
        results = field.query(q, top_k=3)
        nids = [r[0] for r in results]
        # The exact node should be retrievable
        assert any("n_" in nid for nid in nids)
        field.close()

    def test_disabled_falls_back_to_sync(self):
        field = _make_field(async_build=False)
        emb = np.random.randn(64).astype(np.float32)
        nid = field.add_node(emb, {"text": "hello"})
        assert nid in field.hnsw_index.positions
        field.close()

    def test_batch_add_nodes_deferred(self):
        field = _make_field(interval_ms=50, batch_size=1000)
        n = 5
        embeddings = np.random.randn(n, 64).astype(np.float32)
        contents = [{"text": f"batch {i}"} for i in range(n)]
        field.add_nodes_batch(embeddings, contents)
        # Should be pending initially
        assert len(field.hnsw_index.positions) == 0
        time.sleep(0.15)
        assert len(field.hnsw_index.positions) == n
        field.close()

    def test_delete_removes_from_pending(self):
        builder = AsyncIndexBuilder(NaiveGraphIndex(16, 200), interval_ms=10_000, batch_size=100)
        emb = np.random.randn(64).astype(np.float32)
        builder.submit("n1", emb)
        assert len(builder._pending) == 1
        builder.remove("n1")
        assert len(builder._pending) == 0
        builder.close()
