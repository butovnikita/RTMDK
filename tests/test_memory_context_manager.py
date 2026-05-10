"""Tests for RTMDKMemory resource cleanup (close / context manager)."""

import numpy as np

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def _embedder(text: str):
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(64).astype(np.float32)


class TestMemoryResourceCleanup:
    def test_context_manager_closes_on_exit(self):
        cfg = RTMDKConfig(latent_dim=64, use_hnsw=False, wal_fsync_interval_ms=0)
        with RTMDKMemory(config=cfg, embedder=_embedder) as mem:
            mem.add_node(
                content={"text": "hello", "topic": "test"},
                embedding=np.random.randn(64).astype(np.float32),
            )
            assert len(mem.field.nodes) == 1
        # After exit, WAL should be closed (no exception)

    def test_explicit_close(self):
        cfg = RTMDKConfig(latent_dim=64, use_hnsw=False, wal_fsync_interval_ms=0)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        mem.add_node(
            content={"text": "hello", "topic": "test"},
            embedding=np.random.randn(64).astype(np.float32),
        )
        mem.close()
        # Safe to call twice
        mem.close()

    def test_context_manager_exception_still_closes(self):
        cfg = RTMDKConfig(latent_dim=64, use_hnsw=False, wal_fsync_interval_ms=0)
        try:
            with RTMDKMemory(config=cfg, embedder=_embedder) as mem:
                mem.add_node(
                    content={"text": "hello", "topic": "test"},
                    embedding=np.random.randn(64).astype(np.float32),
                )
                raise ValueError("intentional")
        except ValueError:
            pass
        # Should not leak resources
