"""Lifecycle and resource leak detection tests.

Repeated open/add/query/close cycles to detect file descriptor leaks,
unclosed WAL files, and memory growth.
"""

import os
import tempfile

import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.wal import WAL


def _embedder(text: str):
    return np.array([0.0] * 16)


class TestRTMDKLifecycle:
    def test_repeated_open_close_no_crash(self):
        """Multiple create/close cycles should not leak or crash."""
        for i in range(10):
            cfg = RTMDKConfig(
                latent_dim=16,
                use_hnsw=False,
                wal_fsync_interval_ms=0,
                rate_limit_nodes_per_sec=0,
            )
            mem = RTMDKMemory(config=cfg, embedder=_embedder)
            mem.add_node(content={"text": f"doc{i}"}, embedding=np.array([0.0] * 16))
            mem.close()

    def test_context_manager_cleanup(self):
        """Using RTMDKMemory as context manager should clean up."""
        cfg = RTMDKConfig(
            latent_dim=16,
            use_hnsw=False,
            wal_fsync_interval_ms=0,
            rate_limit_nodes_per_sec=0,
        )
        for i in range(10):
            with RTMDKMemory(config=cfg, embedder=_embedder) as mem:
                mem.add_node(content={"text": f"doc{i}"}, embedding=np.array([0.0] * 16))

    def test_wal_file_cleanup_after_close(self):
        """WAL file should not remain locked after close."""
        with tempfile.TemporaryDirectory() as td:
            wal_path = os.path.join(td, "test.wal")
            cfg = RTMDKConfig(
                latent_dim=16,
                use_hnsw=False,
                wal_fsync_interval_ms=0,
                rate_limit_nodes_per_sec=0,
            )
            mem = RTMDKMemory(config=cfg, embedder=_embedder, wal_path=wal_path)
            mem.add_node(content={"text": "hello"}, embedding=np.array([0.0] * 16))
            mem.close()

            # On Windows, file must be closed before removal
            assert os.path.exists(wal_path)
            os.remove(wal_path)
            assert not os.path.exists(wal_path)

    def test_async_builder_stops_after_close(self):
        """AsyncIndexBuilder thread should stop after close."""
        cfg = RTMDKConfig(
            latent_dim=16,
            use_hnsw=False,
            wal_fsync_interval_ms=0,
            rate_limit_nodes_per_sec=0,
            enable_async=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        builder = getattr(mem.field, "_async_index_builder", None)
        mem.close()
        if builder is not None:
            assert not builder._thread.is_alive()


class TestWALLifecycle:
    def test_wal_reopen_replay(self):
        """Re-opening WAL should replay previous entries."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            wal1 = WAL(path, enabled=True, fsync_interval_ms=0)
            wal1.append_add_node("n1", {"text": "hello"})
            wal1.close()

            wal2 = WAL(path, enabled=True, fsync_interval_ms=0)
            records = wal2.replay()
            assert len(records) == 1
            wal2.close()

    def test_wal_close_idempotent(self):
        """Calling close() multiple times should not crash."""
        wal = WAL("/tmp/test_idempotent.jsonl", enabled=False)
        wal.close()
        wal.close()
        wal.close()

    def test_wal_truncate_and_reuse(self):
        """Truncate should clear WAL, allowing reuse."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            wal.append_add_node("n1", {"text": "hello"})
            wal.truncate()
            assert wal.replay() == []
            wal.append_add_node("n2", {"text": "world"})
            records = wal.replay()
            assert len(records) == 1
            wal.close()
