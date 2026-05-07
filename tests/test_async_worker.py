"""Tests for async save worker / background ingestion (Track 6)."""

import asyncio
import numpy as np
from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestAsyncWorker:
    def test_queue_add_nodes_sync_fallback(self, tmp_path):
        """When async_pipeline=False queue_add_nodes falls back to sync."""
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64, use_hnsw=False, hyperbolic=False,
            quantization="none", async_pipeline=False, enable_engrams=False,
        )
        embedder = _make_embedder(64)
        mem = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)
        n = 3
        embeddings = np.random.randn(n, 64).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        mem.field.queue_add_nodes(embeddings, contents)
        assert len(mem.field.nodes) == n

    def test_queue_add_nodes_async(self, tmp_path):
        """Async pipeline: queue_add_nodes + worker eventually ingests."""
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64, use_hnsw=False, hyperbolic=False,
            quantization="none", async_pipeline=True, enable_engrams=False,
        )
        embedder = _make_embedder(64)
        mem = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)

        n = 5
        embeddings = np.random.randn(n, 64).astype(np.float32)
        contents = [{"text": f"async {i}"} for i in range(n)]
        mem.field.queue_add_nodes(embeddings, contents)

        # Start workers manually (no running loop in sync test)
        async def _run():
            await mem.field._start_workers()
            for _ in range(50):
                if len(mem.field.nodes) == n:
                    return True
                await asyncio.sleep(0.05)
            return False

        result = asyncio.run(_run())
        assert result, f"Expected {n} nodes, got {len(mem.field.nodes)}"

    def test_queue_add_nodes_backpressure(self, tmp_path):
        """When queue is full, fallback to synchronous path."""
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64, use_hnsw=False, hyperbolic=False,
            quantization="none", async_pipeline=True, save_queue_size=1,
            enable_engrams=False,
        )
        embedder = _make_embedder(64)
        mem = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)

        # Fill the queue without consuming
        mem.field.save_q.put_nowait({"embeddings": np.zeros(
            (1, 64), dtype=np.float32), "contents": [{"text": "filler"}]})

        n = 3
        embeddings = np.random.randn(n, 64).astype(np.float32)
        contents = [{"text": f"backpressure {i}"} for i in range(n)]
        # Should not raise; falls back to sync
        mem.field.queue_add_nodes(embeddings, contents)
        assert len(mem.field.nodes) == n
