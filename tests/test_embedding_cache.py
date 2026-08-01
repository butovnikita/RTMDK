"""Tests for rtmdk.production.embedding_cache."""

import numpy as np
import pytest

from rtmdk.production.embedding_cache import EmbeddingCache


def _embedder(text: str) -> np.ndarray:
    h = hash(text) % (2**32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(64, dtype=np.float32)


class TestEmbeddingCache:
    def test_get_or_compute_miss_and_hit(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path / "emb_cache"), max_size=10)
        emb1 = cache.get_or_compute("hello", _embedder)
        assert isinstance(emb1, np.ndarray)
        assert emb1.shape == (64,)
        # Second call should be a hit
        emb2 = cache.get_or_compute("hello", _embedder)
        assert np.allclose(emb1, emb2)
        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] == 1

    def test_get_or_compute_batch(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path / "emb_cache2"), max_size=10)
        texts = ["a", "b", "c"]
        results = cache.get_or_compute_batch(texts, _embedder)
        assert len(results) == 3
        assert all(isinstance(r, np.ndarray) for r in results)
        # Second batch should hit cache
        results2 = cache.get_or_compute_batch(texts, _embedder)
        assert all(np.allclose(r1, r2) for r1, r2 in zip(results, results2))

    def test_memory_cache_eviction(self, tmp_path):
        cache = EmbeddingCache(
            cache_dir=str(tmp_path / "emb_cache3"),
            max_size=100,
            memory_cache_size=2,
        )
        cache.get_or_compute("a", _embedder)
        cache.get_or_compute("b", _embedder)
        cache.get_or_compute("c", _embedder)
        assert len(cache.memory_cache) == 2

    def test_disk_persistence(self, tmp_path):
        dir_path = str(tmp_path / "emb_cache4")
        cache = EmbeddingCache(cache_dir=dir_path, max_size=10)
        emb = cache.get_or_compute("persist", _embedder)
        # Create new cache instance pointing to same dir
        cache2 = EmbeddingCache(cache_dir=dir_path, max_size=10)
        emb2 = cache2.get_or_compute("persist", _embedder)
        assert np.allclose(emb, emb2)
        assert cache2.get_stats()["hits"] >= 1

    def test_clear(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path / "emb_cache5"), max_size=10)
        cache.get_or_compute("x", _embedder)
        cache.clear()
        assert len(cache.memory_cache) == 0
        assert cache.get_stats()["hits"] == 0

    def test_hit_rate(self, tmp_path):
        cache = EmbeddingCache(cache_dir=str(tmp_path / "emb_cache6"), max_size=10)
        cache.get_or_compute("rate", _embedder)
        cache.get_or_compute("rate", _embedder)
        assert cache.hit_rate == pytest.approx(0.5, 0.01)
