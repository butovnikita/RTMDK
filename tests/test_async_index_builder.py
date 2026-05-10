"""Unit tests for AsyncIndexBuilder (background HNSW indexing)."""

import time

import numpy as np

from rtmdk.memory.async_index import AsyncIndexBuilder


class _MockHnswIndex:
    def __init__(self):
        self.inserted = {}
        self.batch_calls = 0

    def insert(self, node_id: str, latent: np.ndarray) -> None:
        self.inserted[node_id] = latent.copy()

    def insert_batch(self, node_ids, latents) -> None:
        self.batch_calls += 1
        for nid, lat in zip(node_ids, latents):
            self.inserted[nid] = lat.copy()

    def remove(self, node_id: str) -> None:
        self.inserted.pop(node_id, None)


class TestAsyncIndexBuilder:
    def test_submit_inserts_on_flush(self):
        hnsw = _MockHnswIndex()
        builder = AsyncIndexBuilder(hnsw, interval_ms=10_000, batch_size=2)

        builder.submit("n1", np.array([1.0, 2.0], dtype=np.float32))
        builder.submit("n2", np.array([3.0, 4.0], dtype=np.float32))
        # batch_size=2 triggers immediate flush
        assert len(hnsw.inserted) == 2
        assert "n1" in hnsw.inserted
        builder.close()

    def test_flush_manual(self):
        hnsw = _MockHnswIndex()
        builder = AsyncIndexBuilder(hnsw, interval_ms=10_000, batch_size=100)

        builder.submit("n1", np.array([1.0, 2.0], dtype=np.float32))
        assert len(hnsw.inserted) == 0
        builder.flush()
        assert len(hnsw.inserted) == 1
        builder.close()

    def test_background_flush(self):
        hnsw = _MockHnswIndex()
        builder = AsyncIndexBuilder(hnsw, interval_ms=50, batch_size=100)

        builder.submit("n1", np.array([1.0, 2.0], dtype=np.float32))
        assert len(hnsw.inserted) == 0
        time.sleep(0.1)
        assert len(hnsw.inserted) == 1
        builder.close()

    def test_remove_from_pending(self):
        hnsw = _MockHnswIndex()
        builder = AsyncIndexBuilder(hnsw, interval_ms=10_000, batch_size=100)

        builder.submit("n1", np.array([1.0, 2.0], dtype=np.float32))
        builder.remove("n1")
        builder.flush()
        assert "n1" not in hnsw.inserted
        builder.close()

    def test_close_flushes_remaining(self):
        hnsw = _MockHnswIndex()
        builder = AsyncIndexBuilder(hnsw, interval_ms=10_000, batch_size=100)

        builder.submit("n1", np.array([1.0, 2.0], dtype=np.float32))
        assert len(hnsw.inserted) == 0
        builder.close()
        assert len(hnsw.inserted) == 1

    def test_submit_batch(self):
        hnsw = _MockHnswIndex()
        builder = AsyncIndexBuilder(hnsw, interval_ms=10_000, batch_size=5)

        nids = [f"n{i}" for i in range(3)]
        latents = np.random.randn(3, 64).astype(np.float32)
        builder.submit_batch(nids, latents)
        assert len(hnsw.inserted) == 0  # below batch_size
        builder.flush()
        assert len(hnsw.inserted) == 3
        builder.close()

    def test_max_pending_triggers_flush(self):
        hnsw = _MockHnswIndex()
        builder = AsyncIndexBuilder(hnsw, interval_ms=10_000, batch_size=100, max_pending=2)

        builder.submit("n1", np.array([1.0, 2.0], dtype=np.float32))
        builder.submit("n2", np.array([3.0, 4.0], dtype=np.float32))
        # max_pending=2 triggers immediate flush
        assert len(hnsw.inserted) == 2
        builder.close()
