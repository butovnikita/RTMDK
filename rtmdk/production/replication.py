"""rtmdk/production/replication.py — Multi-master replication with WAL.

Light-weight gossip-style replication over HTTP.  Each node maintains a
local WAL (SQLite) and broadcasts mutations to peers.  Conflict resolution
uses per-node logical clocks (Lamport scalar clocks).

Future: Raft consensus for strong consistency (v8.3 roadmap).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WAL store
# ---------------------------------------------------------------------------


class _WALStore:
    """Thread-safe SQLite WAL for mutations."""

    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS wal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clock INTEGER NOT NULL,
                    origin TEXT NOT NULL,
                    mutation TEXT NOT NULL,
                    ts REAL DEFAULT (julianday('now'))
                )
                """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_wal_clock ON wal(clock)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_wal_origin ON wal(origin)")

    def append(self, clock: int, origin: str, mutation: Dict[str, Any]) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO wal (clock, origin, mutation) VALUES (?, ?, ?)",
                (clock, origin, json.dumps(mutation)),
            )
            return int(cur.lastrowid or 0)

    def since(self, clock: int) -> List[Tuple[int, str, Dict[str, Any]]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT clock, origin, mutation FROM wal WHERE clock > ? ORDER BY clock",
                (clock,),
            )
            return [(c, o, json.loads(m)) for c, o, m in cur.fetchall()]

    def max_clock(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT MAX(clock) FROM wal")
            row = cur.fetchone()
            return int(row[0]) if row and row[0] else 0

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM wal")
            return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# Replication manager
# ---------------------------------------------------------------------------


class ReplicationManager:
    """Light-weight multi-master replication manager.

    Args:
        peers: List of peer base URLs (e.g. ["http://node-b:8000"]).
        node_id: Unique identifier for this node.
        wal_path: Path to local WAL SQLite file (default: in-memory).
        http_timeout: Seconds to wait for peer HTTP responses.
    """

    def __init__(
        self,
        peers: Optional[List[str]] = None,
        node_id: str = "node_1",
        wal_path: Optional[str] = None,
        http_timeout: float = 5.0,
    ):
        self.peers = [p.rstrip("/") for p in (peers or [])]
        self.node_id = node_id
        self.http_timeout = http_timeout
        self._enabled = bool(self.peers)
        self._wal = _WALStore(wal_path or ":memory:")
        self._local_clock = 0
        self._clock_lock = threading.Lock()
        self._httpx: Any = None
        if self._enabled:
            try:
                import httpx

                self._httpx = httpx
            except ImportError as exc:
                logger.warning("httpx not installed; replication disabled (%s)", exc)
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _bump_clock(self) -> int:
        with self._clock_lock:
            self._local_clock += 1
            return self._local_clock

    # -- public API --------------------------------------------------------

    def replicate(self, mutation: Dict[str, Any]) -> bool:
        """Append mutation to local WAL and broadcast to peers.

        Returns ``True`` if local append succeeded (peer failures are logged
        but not treated as fatal).
        """
        if not self._enabled:
            return False
        clock = self._bump_clock()
        stamped = {
            **mutation,
            "_rep_clock": clock,
            "_rep_origin": self.node_id,
            "_rep_ts": time.time(),
        }
        self._wal.append(clock, self.node_id, stamped)
        self._broadcast(stamped)
        return True

    def sync_from_peers(self) -> List[Dict[str, Any]]:
        """Pull missed mutations from all peers.

        Returns a flat list of mutations ordered by logical clock.
        """
        if not self._enabled or self._httpx is None:
            return []
        local_max = self._wal.max_clock()
        merged: Dict[int, Dict[str, Any]] = {}
        for peer in self.peers:
            try:
                with self._httpx.Client(timeout=self.http_timeout) as client:
                    resp = client.get(
                        f"{peer}/v1/replication/wal",
                        params={"since": local_max},
                    )
                    resp.raise_for_status()
                    payload = resp.json()
                    for item in payload.get("mutations", []):
                        clock = item.get("_rep_clock", 0)
                        merged[clock] = item
            except Exception as exc:
                logger.warning("Sync from %s failed: %s", peer, exc)
        # Append to local WAL
        for clock in sorted(merged):
            item = merged[clock]
            self._wal.append(
                clock,
                item.get("_rep_origin", "unknown"),
                item,
            )
            if clock > self._local_clock:
                with self._clock_lock:
                    self._local_clock = clock
        return [merged[c] for c in sorted(merged) if c > local_max]

    def get_wal(self, since: int = 0) -> List[Dict[str, Any]]:
        """Return local WAL entries with clock > *since*."""
        rows = self._wal.since(since)
        return [{**mut, "_rep_clock": clock, "_rep_origin": origin} for clock, origin, mut in rows]

    def local_clock(self) -> int:
        return self._local_clock

    # -- internal ----------------------------------------------------------

    def _broadcast(self, mutation: Dict[str, Any]) -> None:
        if self._httpx is None:
            return
        for peer in self.peers:
            last_exc = None
            for attempt in range(3):
                try:
                    with self._httpx.Client(timeout=self.http_timeout) as client:
                        resp = client.post(
                            f"{peer}/v1/replication/mutation",
                            json=mutation,
                        )
                        resp.raise_for_status()
                    break
                except Exception as exc:
                    last_exc = exc
                    __import__("time").sleep(0.1 * (2**attempt))
            else:
                logger.warning("Replication to %s failed after 3 retries: %s", peer, last_exc)


# ---------------------------------------------------------------------------
# Convenience alias for backward compatibility
# ---------------------------------------------------------------------------

# Backward-compatible alias from older stub
Replicator = ReplicationManager
