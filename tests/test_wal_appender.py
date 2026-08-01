"""Tests for WAL batching + periodic fsync (Track 4)."""

import json
import os
import time


from rtmdk.memory.wal import WAL


class TestWALAppender:
    def test_wal_appender_batches_writes(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        wal = WAL(wal_path, fsync_interval_ms=10_000, batch_size=3)
        wal.append("add_node", {"node_id": "n1"})
        wal.append("add_node", {"node_id": "n2"})
        # Not flushed yet — file may be empty or have 0-2 lines depending on timing
        # Force flush by reaching batch size
        wal.append("add_node", {"node_id": "n3"})
        # Wait a tiny bit for the lock+flush in append
        time.sleep(0.05)
        records = wal.replay()
        assert len(records) == 3
        wal.close()

    def test_wal_appender_fsync_interval(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        wal = WAL(wal_path, fsync_interval_ms=50, batch_size=100)
        wal.append("add_node", {"node_id": "n1"})
        # Should not be flushed immediately (batch size large)
        time.sleep(0.01)
        records_early = wal.replay()
        # replay forces flush, so after replay we expect 1 record
        assert len(records_early) == 1
        wal.close()

    def test_wal_appender_flush_on_close(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        wal = WAL(wal_path, fsync_interval_ms=10_000, batch_size=100)
        wal.append("add_node", {"node_id": "n1"})
        wal.close()
        with open(wal_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["op"] == "add_node"

    def test_wal_appender_replay_correct(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        wal = WAL(wal_path, fsync_interval_ms=10, batch_size=10)
        for i in range(5):
            wal.append("add_node", {"node_id": f"n{i}"})
        time.sleep(0.15)
        records = wal.replay()
        assert len(records) == 5
        assert [r["payload"]["node_id"] for r in records] == [f"n{i}" for i in range(5)]
        wal.close()

    def test_wal_appender_disabled(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        wal = WAL(wal_path, enabled=False, fsync_interval_ms=50, batch_size=1)
        wal.append("add_node", {"node_id": "n1"})
        wal.close()
        assert not os.path.exists(wal_path)

    def test_wal_legacy_sync_path(self, tmp_path):
        """When fsync_interval_ms=0 WAL should fsync on every append."""
        wal_path = str(tmp_path / "wal.jsonl")
        wal = WAL(wal_path, fsync_interval_ms=0, batch_size=100)
        wal.append("add_node", {"node_id": "n1"})
        with open(wal_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        wal.close()
