"""rtmdk/production/audit_log.py — Audit logging for memory operations.

Tracks who did what, when, and from where.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AuditRecord:
    """Single audit log entry."""

    timestamp: float
    action: str  # create_node, update_node, delete_node, query, etc.
    actor: str  # tenant_id or "anonymous"
    resource: str  # node_id, query string, etc.
    details: Dict = field(default_factory=dict)
    client_ip: Optional[str] = None
    request_id: Optional[str] = None


class AuditLog:
    """Persistent audit log with JSON-lines storage.

    Usage:
        log = AuditLog()
        log.record("create_node", actor="t1", resource="n1", details={"content": "hello"})
    """

    def __init__(self, storage_path: Optional[str] = None, max_entries: int = 100000):
        if storage_path is None:
            storage_path = str(Path.home() / ".rtmdk" / "audit.log.jsonl")
        self.storage_path = storage_path
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._buffer: List[AuditRecord] = []
        self._lock = False  # Simple lock flag for single-threaded asyncio

    def record(
        self,
        action: str,
        actor: str = "anonymous",
        resource: str = "",
        details: Optional[Dict] = None,
        client_ip: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Record an audit entry."""
        rec = AuditRecord(
            timestamp=time.time(),
            action=action,
            actor=actor,
            resource=resource,
            details=details or {},
            client_ip=client_ip,
            request_id=request_id,
        )
        self._buffer.append(rec)
        if len(self._buffer) >= 100:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        lines = []
        for rec in self._buffer:
            lines.append(json.dumps(asdict(rec), ensure_ascii=False))
        with open(self.storage_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        self._buffer.clear()
        self._rotate_if_needed()

    def _rotate_if_needed(self):
        """Simple rotation: truncate if file too large."""
        try:
            if os.path.getsize(self.storage_path) > self.max_entries * 500:
                # Keep last 50% of entries
                with open(self.storage_path, "r", encoding="utf-8") as fh:
                    lines = fh.readlines()
                keep = lines[-(self.max_entries // 2):]
                with open(self.storage_path, "w", encoding="utf-8") as fh:
                    fh.writelines(keep)
        except Exception:
            pass

    def query(
        self,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Query audit log entries."""
        self._flush()
        results: List[Dict] = []
        if not os.path.exists(self.storage_path):
            return results
        try:
            with open(self.storage_path, "r", encoding="utf-8") as fh:
                for line in reversed(fh.readlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if actor is not None and rec.get("actor") != actor:
                        continue
                    if action is not None and rec.get("action") != action:
                        continue
                    if since is not None and rec.get("timestamp", 0) < since:
                        continue
                    results.append(rec)
                    if len(results) >= limit:
                        break
        except Exception:
            pass
        return results

    def close(self):
        self._flush()
