"""Tests for rtmdk.support.hnsw (NaiveGraphIndex / HNSWIndex alias)."""
import numpy as np

from rtmdk.support.hnsw import NaiveGraphIndex, HNSWIndex


class TestNaiveGraphIndex:
    def test_insert_and_search(self):
        idx = NaiveGraphIndex(m=2, ef_construction=10)
        positions = {
            "a": np.array([0.0, 0.0], dtype=np.float32),
            "b": np.array([1.0, 0.0], dtype=np.float32),
            "c": np.array([0.0, 1.0], dtype=np.float32),
        }
        for nid, pos in positions.items():
            idx.insert(nid, pos)
        results = idx.search(np.array([0.1, 0.1], dtype=np.float32), top_k=2)
        assert len(results) == 2
        assert "a" in results  # closest to query

    def test_remove(self):
        idx = NaiveGraphIndex(m=2)
        idx.insert("x", np.array([0.0, 0.0], dtype=np.float32))
        idx.insert("y", np.array([1.0, 0.0], dtype=np.float32))
        idx.remove("x")
        results = idx.search(np.array([0.0, 0.0], dtype=np.float32), top_k=1)
        assert results == ["y"]

    def test_search_empty_returns_empty(self):
        idx = NaiveGraphIndex()
        assert idx.search(np.array([0.0, 0.0], dtype=np.float32)) == []

    def test_backward_compatible_alias(self):
        """HNSWIndex must still work as a class alias."""
        idx = HNSWIndex(m=2, ef_construction=5)
        idx.insert("old", np.array([0.0, 0.0], dtype=np.float32))
        assert isinstance(idx, NaiveGraphIndex)
