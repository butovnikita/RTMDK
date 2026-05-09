"""Unit tests for IndexManager (HNSW + BM25 + shard routing)."""
import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.index_manager import IndexManager
from rtmdk.memory.quantization import QuantizationHelper


def _make_cfg(**kwargs):
    defaults = dict(
        latent_dim=64,
        use_hnsw=True,
        bm25_fallback=True,
        sparse_routing=True,
        num_shards=4,
        bm25_k1=1.5,
        bm25_b=0.75,
    )
    defaults.update(kwargs)
    return RTMDKConfig(**defaults)


def test_index_manager_hnsw_insert_and_search():
    cfg = _make_cfg(hnsw_min_nodes=0)
    rng = np.random.default_rng(42)
    quant = QuantizationHelper("none")
    mgr = IndexManager(cfg, 64, rng, quant)

    assert mgr.hnsw_count() == 0
    vec = rng.standard_normal(64).astype(np.float32)
    mgr.hnsw_insert("n1", vec)
    assert mgr.hnsw_count() == 1

    results = mgr.hnsw_search(vec, top_k=5)
    assert "n1" in results


def test_index_manager_bm25_add_and_search():
    cfg = _make_cfg()
    rng = np.random.default_rng(42)
    quant = QuantizationHelper("none")
    mgr = IndexManager(cfg, 64, rng, quant)

    mgr.bm25_add("d1", "hello world memory neural search")
    mgr.bm25_add("d2", "vector embedding space cosine similarity")

    results = mgr.bm25_search("neural search", top_k=5)
    ids = [r[0] for r in results]
    assert "d1" in ids


def test_index_manager_shard_routing():
    cfg = _make_cfg()
    rng = np.random.default_rng(42)
    quant = QuantizationHelper("none")
    mgr = IndexManager(cfg, 64, rng, quant)

    query = rng.standard_normal(64).astype(np.float32)
    shards = mgr.route_query(query, top_shards=2)
    assert len(shards) == 2
    assert all(0 <= s < 4 for s in shards)


def test_index_manager_shard_centers_update():
    cfg = _make_cfg()
    rng = np.random.default_rng(42)
    quant = QuantizationHelper("none")
    mgr = IndexManager(cfg, 64, rng, quant)

    # Fake nodes
    class FakeNode:
        def __init__(self, pos):
            self.latent_pos = pos

    nodes = {f"n{i}": FakeNode(rng.standard_normal(64).astype(np.float32)) for i in range(20)}
    node_index = list(nodes.keys())

    mgr.update_shard_centers(nodes, node_index, cfg.num_shards)
    assert mgr.shard_centers is not None
    assert mgr.shard_centers.shape == (4, 64)


def test_index_manager_get_node_shard():
    cfg = _make_cfg()
    rng = np.random.default_rng(42)
    quant = QuantizationHelper("none")
    mgr = IndexManager(cfg, 64, rng, quant)

    class FakeNode:
        def __init__(self, pos):
            self.latent_pos = pos

    node = FakeNode(rng.standard_normal(64).astype(np.float32))
    shard = mgr.get_node_shard("n1", node)
    assert 0 <= shard < 4
