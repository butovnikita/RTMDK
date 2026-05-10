"""Unit tests for WAL (Write-Ahead Log)."""

import os
import tempfile
import time

from rtmdk.memory.wal import WAL


class TestWAL:
    def test_sync_append_replay(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            wal.append_add_node("n1", {"text": "hello"}, embedding=[0.1, 0.2])
            wal.append_delete(["n2"])
            records = wal.replay()
            assert len(records) == 2
            assert records[0]["op"] == "add_node"
            assert records[0]["payload"]["node_id"] == "n1"
            assert records[1]["op"] == "delete"
            wal.close()

    def test_async_append_flush_on_close(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            wal = WAL(path, enabled=True, fsync_interval_ms=10_000, batch_size=100)
            wal.append_add_node("n1", {"text": "hello"})
            wal.append_add_node("n2", {"text": "world"})
            # Buffer should hold data; close must flush
            wal.close()
            wal2 = WAL(path, enabled=True, fsync_interval_ms=0)
            records = wal2.replay()
            assert len(records) == 2
            wal2.close()

    def test_async_batch_flush(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            wal = WAL(path, enabled=True, fsync_interval_ms=10_000, batch_size=2)
            wal.append_add_node("n1", {"text": "a"})
            wal.append_add_node("n2", {"text": "b"})
            # batch_size=2 triggers flush
            time.sleep(0.05)
            wal2 = WAL(path, enabled=True, fsync_interval_ms=0)
            records = wal2.replay()
            assert len(records) == 2
            wal2.close()
            wal.close()

    def test_truncate_clears_wal(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            wal.append_add_node("n1", {"text": "hello"})
            wal.truncate()
            records = wal.replay()
            assert records == []
            wal.close()

    def test_disabled_wal_noop(self):
        wal = WAL("/tmp/should_not_create.jsonl", enabled=False)
        wal.append_add_node("n1", {"text": "hello"})
        assert wal.replay() == []
        wal.close()

    def test_append_consolidate(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            wal.append_consolidate(["n1", "n2"])
            records = wal.replay()
            assert len(records) == 1
            assert records[0]["op"] == "consolidate"
            assert records[0]["payload"]["updated"] == ["n1", "n2"]
            wal.close()
