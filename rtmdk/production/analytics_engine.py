"""
rtmdk/production/analytics_engine.py — Analytics Tracking Engine v2.

Structured event taxonomy with persistent storage.
Built for decision-grade analytics on the RTMDK memory system.
"""

import json
import time
import uuid
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from contextlib import contextmanager
from collections import defaultdict
import threading


# ============================================================================
# EVENT TAXONOMY
# ============================================================================
# Every event follows: object_action[_context]
# Properties carry context, never PII or free-text.

class EventType:
    # Navigation / Exposure
    QUERY_RECEIVED = "query_received"           # API query entered system
    MEMORY_ACCESSED = "memory_accessed"         # Memory field was accessed

    # Intent Signals
    CONTEXT_INJECTED = "context_injected"         # Memory context injected into response
    IMAGINE_TRIGGERED = "imagine_triggered"      # Counterfactual imagination activated

    # Completion Signals
    CONSOLIDATION_COMPLETED = "consolidation_completed"  # Memory consolidation finished
    NODE_CREATED = "node_created"               # New memory node created
    NODE_PRUNED = "node_pruned"                 # Memory node removed

    # System / State Changes
    FIELD_HEALTH_CHANGED = "field_health_changed"  # Field health state transition
    ERROR_OCCURRED = "error_occurred"           # System error intercepted


# ============================================================================
# ANALYTICS STORE (SQLite-backed)
# ============================================================================

class AnalyticsStore:
    """
    Persistent analytics store with SQLite.
    Thread-safe, auto-creating schema.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path.home() / ".rtmdk" / "analytics.db"
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id          TEXT PRIMARY KEY,
                    event_type  TEXT NOT NULL,
                    user_id     TEXT,
                    session_id  TEXT,
                    properties  TEXT,
                    timestamp   REAL NOT NULL,
                    received_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            """)

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversions (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL UNIQUE,
                    event_type  TEXT NOT NULL,
                    counting    TEXT NOT NULL DEFAULT 'once_per_session',
                    created_at  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversion_fires (
                    id              TEXT PRIMARY KEY,
                    conversion_name TEXT NOT NULL,
                    session_id      TEXT NOT NULL,
                    timestamp       REAL NOT NULL,
                    properties      TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_fires_conversion ON conversion_fires(conversion_name);
                CREATE INDEX IF NOT EXISTS idx_fires_session ON conversion_fires(session_id);
            """)

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def track(self, event_type: str, properties: Dict = None,
              user_id: str = None, session_id: str = None):
        """Persist an event."""
        event = {
            "id": str(uuid.uuid4()),
            "event_type": event_type,
            "user_id": user_id,
            "session_id": session_id,
            "properties": json.dumps(properties or {}),
            "timestamp": time.time(),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO events(id, event_type, user_id, session_id, properties, timestamp, received_at)
                   VALUES(:id, :event_type, :user_id, :session_id, :properties, :timestamp, :received_at)""",
                event
            )
            conn.commit()

    def query(self, event_type: str = None, session_id: str = None,
              since: float = None, limit: int = 1000) -> List[Dict]:
        """Query events with filters."""
        q = "SELECT * FROM events WHERE 1=1"
        params = []
        if event_type:
            q += " AND event_type = ?"
            params.append(event_type)
        if session_id:
            q += " AND session_id = ?"
            params.append(session_id)
        if since:
            q += " AND timestamp >= ?"
            params.append(since)
        q += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(q, params).fetchall()
        return [dict(r) for r in rows]

    def register_conversion(self, name: str, event_type: str,
                           counting: str = "once_per_session"):
        """Register a conversion."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO conversions(name, event_type, counting, created_at)
                   VALUES(?, ?, ?, ?)""",
                (name, event_type, counting, time.time())
            )
            conn.commit()

    def fire_conversion(self, name: str, session_id: str,
                       properties: Dict = None):
        """Fire a conversion, respecting counting rules."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT event_type, counting FROM conversions WHERE name = ?",
                (name,)
            ).fetchone()
            if not row:
                return False

            if row["counting"] == "once_per_session":
                exists = conn.execute(
                    "SELECT 1 FROM conversion_fires WHERE conversion_name = ? AND session_id = ?",
                    (name, session_id)
                ).fetchone()
                if exists:
                    return False  # Already fired

            conn.execute(
                """INSERT INTO conversion_fires(id, conversion_name, session_id, timestamp, properties)
                   VALUES(?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), name, session_id, time.time(),
                 json.dumps(properties or {}))
            )
            conn.commit()
            return True

    def get_conversion_stats(self) -> Dict[str, Dict]:
        """Get conversion counts."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT conversion_name, COUNT(*) as count
                   FROM conversion_fires GROUP BY conversion_name"""
            ).fetchall()
        return {r["conversion_name"]: {"count": r["count"]} for r in rows}


# ============================================================================
# ANALYTICS ENGINE
# ============================================================================

class AnalyticsEngine:
    """
    Central analytics engine for RTMDK.
    Wraps event tracking, conversions, and reporting.

    Usage:
        analytics = AnalyticsEngine()

        # Track an event
        analytics.track("query_received", {"query_type": "recall"}, session_id="s_123")

        # Define conversions
        analytics.define_conversion("memory_context_used", "context_injected")

        # Fire conversion
        analytics.fire_conversion("memory_context_used", session_id="s_123")

        # Get report
        report = analytics.get_report()
    """

    def __init__(self, store: AnalyticsStore = None):
        self.store = store or AnalyticsStore()
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._session_context: Dict[str, Dict] = {}

    def track(self, event_type: str, properties: Dict = None,
              user_id: str = None, session_id: str = None):
        """Track an event and fire any matching conversions."""
        # Inject session context
        if session_id and session_id in self._session_context:
            ctx = self._session_context[session_id]
            if user_id is None and "user_id" in ctx:
                user_id = ctx["user_id"]

        self.store.track(event_type, properties, user_id, session_id)

        # Fire registered conversions for this event type
        with self.store._conn() as conn:
            rows = conn.execute(
                "SELECT name FROM conversions WHERE event_type = ?",
                (event_type,)
            ).fetchall()
        for row in rows:
            self.fire_conversion(row["name"], session_id, properties)

        # Call handlers
        for handler in self._handlers.get(event_type, []):
            try:
                handler({"type": event_type, "properties": properties,
                         "user_id": user_id, "session_id": session_id})
            except Exception:
                pass

    def on(self, event_type: str, handler: Callable):
        """Register a handler for an event type."""
        self._handlers[event_type].append(handler)

    def define_conversion(self, name: str, event_type: str,
                         counting: str = "once_per_session"):
        """Define a conversion tied to an event type."""
        self.store.register_conversion(name, event_type, counting)

    def fire_conversion(self, name: str, session_id: str = None,
                        properties: Dict = None):
        """Fire a named conversion."""
        self.store.fire_conversion(name, session_id or "unknown", properties)

    def set_session_context(self, session_id: str, user_id: str = None,
                             metadata: Dict = None):
        """Set context for a session (call at request start)."""
        self._session_context[session_id] = {
            "user_id": user_id,
            "metadata": metadata or {},
        }

    def get_report(self, since: float = None) -> Dict[str, Any]:
        """Generate analytics report."""
        since_ts = since or (time.time() - 86400)  # Default: last 24h

        # Event counts by type
        with self.store._conn() as conn:
            rows = conn.execute(
                """SELECT event_type, COUNT(*) as count
                   FROM events WHERE timestamp >= ?
                   GROUP BY event_type ORDER BY count DESC""",
                (since_ts,)
            ).fetchall()
        event_counts = {r["event_type"]: r["count"] for r in rows}

        # Top events over time (hourly buckets)
        with self.store._conn() as conn:
            rows = conn.execute(
                """SELECT
                       strftime('%Y-%m-%d %H', datetime(timestamp, 'unixepoch')) as hour,
                       COUNT(*) as count
                   FROM events WHERE timestamp >= ?
                   GROUP BY hour ORDER BY hour DESC LIMIT 24""",
                (since_ts,)
            ).fetchall()
        time_series = [{"hour": r["hour"], "count": r["count"]} for r in rows]

        # Session count
        with self.store._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT session_id) as cnt FROM events WHERE timestamp >= ?",
                (since_ts,)
            ).fetchone()
        unique_sessions = row["cnt"] if row else 0

        # Conversion stats
        conversions = self.store.get_conversion_stats()

        return {
            "period": {
                "since": since_ts,
                "until": time.time(),
                "label": "last_24h" if since is None else "custom",
            },
            "event_counts": event_counts,
            "unique_sessions": unique_sessions,
            "time_series": time_series,
            "conversions": conversions,
            "generated_at": time.time(),
        }

    def get_event_log(self, event_type: str = None, limit: int = 100
                      ) -> List[Dict]:
        """Get recent event log."""
        return self.store.query(event_type=event_type, limit=limit)