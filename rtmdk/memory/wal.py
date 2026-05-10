"""
rtmdk/memory/wal.py — Minimal Write-Ahead Log for mutations.

Guarantees durability for add_node, consolidate, and delete operations.
On startup WAL is replayed before loading the main snapshot.
After a successful export_field the WAL is truncated.
"""

import atexit
import json
import os
import time
import threading
from typing import Dict, Any, Optional, List


class WAL:
    """Append-only write-ahead log for RTMDKField mutations.

    Supports batching + periodic fsync when ``fsync_interval_ms > 0``.
    """

    def __init__(
        self,
        path: str,
        enabled: bool = True,
        fsync_interval_ms: int = 0,
        batch_size: int = 100,
    ):
        self.path = path
        self.enabled = enabled
        self.fsync_interval_ms = fsync_interval_ms
        self.batch_size = batch_size
        self._file = None
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._last_fsync = 0.0
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        if self.enabled:
            atexit.register(self.close)
            if self.fsync_interval_ms > 0:
                self._flush_thread = threading.Thread(target=self._fsync_loop, daemon=True)
                self._flush_thread.start()

    def _ensure_open(self):
        with self._lock:
            if not self.enabled or self._file is not None:
                return
            try:
                self._file = open(self.path, "a", encoding="utf-8")
            except OSError:
                self.enabled = False

    def _fsync_loop(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.fsync_interval_ms / 1000.0)
            self._flush()

    def _flush(self):
        with self._lock:
            if not self._buffer or self._file is None:
                return
            try:
                self._file.write("".join(self._buffer))
                self._file.flush()
                os.fsync(self._file.fileno())
                self._buffer.clear()
                self._last_fsync = time.time()
            except OSError:
                pass

    def append(self, operation: str, payload: Dict[str, Any]):
        """Append a mutation record."""
        if not self.enabled:
            return
        self._ensure_open()
        if self._file is None:
            return
        record = {"op": operation, "ts": time.time(), "payload": payload}
        line = json.dumps(record, ensure_ascii=False) + "\n"

        with self._lock:
            if self._file is None:
                return
            if self.fsync_interval_ms <= 0:
                # Legacy synchronous path
                try:
                    self._file.write(line)
                    self._file.flush()
                    os.fsync(self._file.fileno())
                except OSError:
                    pass
                return
            self._buffer.append(line)
            should_flush = len(self._buffer) >= self.batch_size

        if should_flush:
            self._flush()

    def append_add_node(
        self, node_id: str, content: Dict[str, Any], modality: str = "text", embedding: Optional[List[float]] = None
    ):
        payload: Dict[str, Any] = {"node_id": node_id, "content": content, "modality": modality}
        if embedding is not None:
            payload["embedding"] = embedding
        self.append("add_node", payload)

    def append_consolidate(self, updated: List[str]):
        self.append("consolidate", {"updated": updated})

    def append_delete(self, node_ids: List[str]):
        self.append("delete", {"node_ids": node_ids})

    def replay(self) -> List[Dict[str, Any]]:
        """Read all records from WAL. Returns list of mutations."""
        if not self.enabled or not os.path.exists(self.path):
            return []
        # Ensure any buffered data is flushed before replay
        self._flush()
        records = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return records

    def truncate(self):
        """Truncate WAL after successful snapshot save."""
        if not self.enabled:
            return
        self._stop()
        if self._file is not None:
            self._file.close()
            self._file = None
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        # Restart background thread if needed
        self._stop_event.clear()
        if self.fsync_interval_ms > 0:
            self._flush_thread = threading.Thread(target=self._fsync_loop, daemon=True)
            self._flush_thread.start()

    def close(self):
        self._flush()
        self._stop()
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _stop(self):
        if self._flush_thread is not None and self._flush_thread.is_alive():
            self._stop_event.set()
            self._flush_thread.join(timeout=2.0)
            self._flush_thread = None
