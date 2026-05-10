"""Regression tests for int8 cache storage (4x RAM reduction)."""

import numpy as np

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def _embedder(text: str):
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(64).astype(np.float32)


class TestInt8Cache:
    def test_int8_cache_dtype_is_int8(self):
        cfg = RTMDKConfig(latent_dim=64, quantization="int8", use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)

        for i in range(10):
            emb = np.random.randn(64).astype(np.float32)
            mem.add_node(
                content={"text": f"node {i}", "topic": "test"},
                embedding=emb,
            )

        # Force cache build
        mem.field._cache_mgr.build(mem.field)

        cache = mem.field._cache_mgr
        assert cache.positions is not None
        assert cache.positions.dtype == np.int8
        assert cache.scales is not None
        assert cache.scales.dtype == np.float32
        assert len(cache.scales) == 10

    def test_fp16_cache_dtype_is_fp16(self):
        cfg = RTMDKConfig(latent_dim=64, quantization="fp16", use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)

        for i in range(5):
            emb = np.random.randn(64).astype(np.float32)
            mem.add_node(
                content={"text": f"node {i}", "topic": "test"},
                embedding=emb,
            )

        mem.field._cache_mgr.build(mem.field)
        assert mem.field._cache_mgr.positions.dtype == np.float16
        assert mem.field._cache_mgr.scales is None

    def test_none_cache_dtype_is_float32(self):
        cfg = RTMDKConfig(latent_dim=64, quantization="none", use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)

        for i in range(5):
            emb = np.random.randn(64).astype(np.float32)
            mem.add_node(
                content={"text": f"node {i}", "topic": "test"},
                embedding=emb,
            )

        mem.field._cache_mgr.build(mem.field)
        assert mem.field._cache_mgr.positions.dtype == np.float32
        assert mem.field._cache_mgr.scales is None
