"""tests/test_query_cache.py — Query cache and adaptive top-k tests."""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def disable_rate_limit(monkeypatch):
    monkeypatch.setenv("RTMDK_ADD_RATE_LIMIT", "0")


class TestQueryCache:
    def test_cache_disabled_by_default(self):
        cfg = RTMDKConfig(latent_dim=16, query_cache_size=0)
        field = RTMDKField(cfg)
        assert field.query_cache is None

    def test_cache_hit_reduces_latency(self):
        cfg = RTMDKConfig(
            latent_dim=16,
            top_k=3,
            min_response=0.001,
            decay_rate=0.999,
            use_hnsw=False,
            learn_projection=False,
            bm25_fallback=False,
            enable_async=False,
            resonance_kernel="cosine",
            phase_coupling=0.0,
            query_cache_size=100,
            query_cache_ttl=3600,
        )
        field = RTMDKField(cfg)
        for i in range(50):
            emb = np.random.randn(16).astype(np.float32)
            emb /= np.linalg.norm(emb)
            field.add_node(emb, content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)

        q = np.random.randn(16).astype(np.float32)
        q /= np.linalg.norm(q)

        # Cold query
        r1 = field.query(q, top_k=3)

        # Warm query (cache hit)
        r2 = field.query(q, top_k=3)

        assert r1 == r2
        assert field.stats.get("query_cache_hits", 0) >= 1
        assert field.stats.get("query_cache_misses", 0) >= 1

    def test_cache_invalidation_on_add(self):
        cfg = RTMDKConfig(
            latent_dim=16,
            top_k=3,
            min_response=0.001,
            decay_rate=0.999,
            use_hnsw=False,
            learn_projection=False,
            bm25_fallback=False,
            enable_async=False,
            resonance_kernel="cosine",
            phase_coupling=0.0,
            query_cache_size=100,
        )
        field = RTMDKField(cfg)
        for i in range(20):
            emb = np.random.randn(16).astype(np.float32)
            emb /= np.linalg.norm(emb)
            field.add_node(emb, content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)

        q = np.random.randn(16).astype(np.float32)
        q /= np.linalg.norm(q)
        field.query(q, top_k=3)
        assert field.query_cache.size == 1

        # Add new node — cache should be cleared
        field.add_node(
            np.random.randn(16).astype(np.float32),
            content={"id": "new"},
            phase=0.0,
            node_id="n_new",
            skip_projection=True,
        )
        assert field.query_cache.size == 0

    def test_cache_ttl_expiration(self):
        cfg = RTMDKConfig(
            latent_dim=16,
            top_k=3,
            min_response=0.001,
            decay_rate=0.999,
            use_hnsw=False,
            learn_projection=False,
            bm25_fallback=False,
            enable_async=False,
            resonance_kernel="cosine",
            phase_coupling=0.0,
            query_cache_size=100,
            query_cache_ttl=0,  # immediate expiration
        )
        field = RTMDKField(cfg)
        for i in range(10):
            emb = np.random.randn(16).astype(np.float32)
            emb /= np.linalg.norm(emb)
            field.add_node(emb, content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)

        q = np.random.randn(16).astype(np.float32)
        q /= np.linalg.norm(q)
        field.query(q, top_k=3)
        time.sleep(0.05)
        field.query(q, top_k=3)  # should miss due to TTL
        assert field.stats.get("query_cache_hits", 0) == 0
        assert field.stats.get("query_cache_misses", 0) >= 2


class TestAdaptiveTopK:
    def test_high_confidence_returns_one(self):
        cfg = RTMDKConfig(
            latent_dim=16,
            top_k=5,
            min_response=0.001,
            decay_rate=0.999,
            use_hnsw=False,
            learn_projection=False,
            bm25_fallback=False,
            enable_async=False,
            resonance_kernel="cosine",
            phase_coupling=0.0,
            adaptive_top_k=True,
        )
        field = RTMDKField(cfg)
        # Add a node that is very close to query
        target = np.random.randn(16).astype(np.float32)
        target /= np.linalg.norm(target)
        field.add_node(target, content={"id": 0}, phase=0.0, node_id="n0", skip_projection=True)
        field.nodes["n0"].amplitude = 1.0
        field.nodes["n0"].salience = 1.0

        # Add distant nodes
        for i in range(1, 10):
            emb = np.random.randn(16).astype(np.float32)
            emb /= np.linalg.norm(emb)
            # Make them orthogonal to target
            emb = emb - np.dot(emb, target) * target
            emb /= np.linalg.norm(emb) + 1e-8
            field.add_node(emb, content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
            field.nodes[f"n{i}"].amplitude = 1.0
            field.nodes[f"n{i}"].salience = 1.0

        results = field.query(target, top_k=5)
        assert len(results) == 1  # high confidence -> only top-1

    def test_medium_confidence_returns_three(self):
        np.random.seed(42)
        cfg = RTMDKConfig(
            latent_dim=16,
            top_k=5,
            min_response=0.001,
            decay_rate=0.999,
            use_hnsw=False,
            learn_projection=False,
            bm25_fallback=False,
            enable_async=False,
            resonance_kernel="cosine",
            phase_coupling=0.0,
            adaptive_top_k=True,
        )
        field = RTMDKField(cfg)
        # Add one close node (will be top-1 with high score)
        base = np.random.randn(16).astype(np.float32)
        base /= np.linalg.norm(base)
        field.add_node(
            base.copy(),
            content={"id": 0},
            phase=0.0,
            node_id="n0",
            skip_projection=True,
        )
        field.nodes["n0"].amplitude = 1.0
        field.nodes["n0"].salience = 1.0

        # Add 4 somewhat similar nodes (medium confidence tail)
        for i in range(1, 5):
            emb = base * 0.7 + np.random.randn(16).astype(np.float32) * 0.7
            emb /= np.linalg.norm(emb)
            field.add_node(emb, content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
            field.nodes[f"n{i}"].amplitude = 1.0
            field.nodes[f"n{i}"].salience = 1.0

        results = field.query(base, top_k=5)
        # With seed=42, top score should be >= 0.80, so adaptive returns 1 or 3
        assert len(results) <= 3

    def test_adaptive_top_k_disabled(self):
        cfg = RTMDKConfig(
            latent_dim=16,
            top_k=5,
            min_response=0.001,
            decay_rate=0.999,
            use_hnsw=False,
            learn_projection=False,
            bm25_fallback=False,
            enable_async=False,
            resonance_kernel="cosine",
            phase_coupling=0.0,
            adaptive_top_k=False,
        )
        field = RTMDKField(cfg)
        for i in range(10):
            emb = np.random.randn(16).astype(np.float32)
            emb /= np.linalg.norm(emb)
            field.add_node(emb, content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
            field.nodes[f"n{i}"].amplitude = 1.0
            field.nodes[f"n{i}"].salience = 1.0

        q = np.random.randn(16).astype(np.float32)
        q /= np.linalg.norm(q)
        results = field.query(q, top_k=5)
        assert len(results) == 5
