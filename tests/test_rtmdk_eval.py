"""
tests/test_rtmdk_eval.py — Tests for analytics and evaluation modules.

Covers:
1. AnalyticsStore event tracking
2. AnalyticsStore querying
3. Conversion tracking
"""

import os
import tempfile

from rtmdk.production.analytics_engine import AnalyticsStore, EventType


class TestAnalyticsStore:
    def test_track_and_query_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "analytics.db")
            store = AnalyticsStore(db_path=db_path)

            store.track(
                event_type=EventType.NODE_CREATED,
                properties={"node_id": "n1"},
                session_id="sess_1",
            )
            events = store.query(event_type=EventType.NODE_CREATED)
            assert len(events) == 1
            assert events[0]["event_type"] == EventType.NODE_CREATED
            assert events[0]["properties"]["node_id"] == "n1"

    def test_query_filters_by_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "analytics.db")
            store = AnalyticsStore(db_path=db_path)

            store.track(EventType.QUERY_RECEIVED, session_id="sess_a")
            store.track(EventType.QUERY_RECEIVED, session_id="sess_b")
            events = store.query(session_id="sess_a")
            assert len(events) == 1

    def test_conversion_once_per_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "analytics.db")
            store = AnalyticsStore(db_path=db_path)

            store.register_conversion("signup", EventType.NODE_CREATED)
            assert store.fire_conversion("signup", "sess_1") is True
            assert store.fire_conversion("signup", "sess_1") is False
            assert store.fire_conversion("signup", "sess_2") is True
