"""Fault injection tests for WAL durability edge cases."""

import os
import sys
import tempfile
import threading
import time

import pytest

from rtmdk.memory.wal import WAL


class TestWALCorruption:
    def test_replay_skips_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"op": "add_node", "payload": {}}\n')
                f.write("this is not json\n")
                f.write('{"op": "delete", "payload": {"node_ids": ["n1"]}}\n')
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            records = wal.replay()
            assert len(records) == 2
            assert records[0]["op"] == "add_node"
            assert records[1]["op"] == "delete"
            wal.close()

    def test_replay_skips_truncated_line(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write('{"op": "add_node", "payload": {}}\n')
                f.write('{"op": "delete", "payload": {"node_id')  # truncated
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            records = wal.replay()
            assert len(records) == 1
            wal.close()

    def test_replay_empty_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            open(path, "w").close()
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            assert wal.replay() == []
            wal.close()

    def test_replay_blank_lines(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n\n")
                f.write('{"op": "add_node", "payload": {}}\n')
                f.write("\n")
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            records = wal.replay()
            assert len(records) == 1
            wal.close()


class TestWALConcurrent:
    def test_concurrent_append_and_close(self):
        path = tempfile.mktemp(suffix=".jsonl")
        wal = WAL(path, enabled=True, fsync_interval_ms=0)
        errors = []

        def appender():
            try:
                for i in range(100):
                    wal.append_add_node(f"n{i}", {"text": f"doc{i}"})
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=appender)
        t1.start()
        time.sleep(0.01)
        wal.close()
        t1.join(timeout=5)

        # Should not crash; some records may be lost due to close
        assert not errors
        # Give Windows time to release file handle
        time.sleep(0.2)
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass

    def test_concurrent_append_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            wal = WAL(path, enabled=True, fsync_interval_ms=5)
            errors = []

            def appender():
                try:
                    for i in range(50):
                        wal.append_add_node(f"n{i}", {"text": f"doc{i}"})
                        time.sleep(0.001)
                except Exception as exc:
                    errors.append(exc)

            def replayer():
                try:
                    for _ in range(20):
                        wal.replay()
                        time.sleep(0.003)
                except Exception as exc:
                    errors.append(exc)

            t1 = threading.Thread(target=appender)
            t2 = threading.Thread(target=replayer)
            t1.start()
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)
            wal.close()

            assert not errors


class TestWALReadOnly:
    @pytest.mark.skipif(sys.platform == "win32", reason="chmod read-only dir not supported on Windows")
    def test_append_fails_gracefully_on_readonly_dir(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "wal.jsonl")
            # Create WAL file first
            wal = WAL(path, enabled=True, fsync_interval_ms=0)
            wal.append_add_node("n1", {"text": "hello"})
            wal.close()

            # Make directory read-only
            os.chmod(td, 0o555)
            try:
                wal2 = WAL(path, enabled=True, fsync_interval_ms=0)
                # Append should fail gracefully (enabled becomes False)
                wal2.append_add_node("n2", {"text": "world"})
                # File should not have new data
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                assert len(lines) == 1
                wal2.close()
            finally:
                os.chmod(td, 0o755)
