"""
rtmdk/memory/wal.py — Minimal Write-Ahead Log for mutations.

Guarantees durability for add_node, consolidate, and delete operations.
On startup WAL is replayed before loading the main snapshot.
After a successful export_field the WAL is truncated.
"""
import json
import os
import time
from typing import Dict, Any, Optional, List


class WAL:
    """Append-only write-ahead log for RTMDKField mutations."""

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._file = None

    def _ensure_open(self):
        if not self.enabled or self._file is not None:
            return
        try:
            self._file = open(self.path, "a", encoding="utf-8")
        except OSError:
            self.enabled = False

    def append(self, operation: str, payload: Dict[str, Any]):
        """Append a mutation record and fsync."""
        if not self.enabled:
            return
        self._ensure_open()
        if self._file is None:
            return
        record = {"op": operation, "ts": time.time(), "payload": payload}
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def append_add_node(self,
                        node_id: str,
                        content: Dict[str,
                                      Any],
                        modality: str = "text",
                        embedding: Optional[List[float]] = None):
        payload = {
            "node_id": node_id,
            "content": content,
            "modality": modality}
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
        if self._file is not None:
            self._file.close()
            self._file = None
        if os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
