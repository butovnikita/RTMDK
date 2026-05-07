"""Tests for rtmdk.production.events."""

import time
from unittest.mock import patch

import pytest
from rtmdk.production.events import EventSystem


class TestEventSystem:
    def test_on_and_emit(self):
        events = EventSystem()
        received = []
        events.on("test", lambda e: received.append(e))
        events.emit("test", {"key": "value"})
        assert len(received) == 1
        assert received[0]["name"] == "test"
        assert received[0]["data"] == {"key": "value"}

    def test_emit_no_handlers(self):
        events = EventSystem()
        events.emit("noop", {})  # should not raise

    def test_add_webhook(self):
        events = EventSystem()
        events.add_webhook("https://example.com/hook", ["node_added"])
        assert len(events._webhooks) == 1

    def test_get_event_log(self):
        events = EventSystem()
        events.emit("a", {})
        events.emit("b", {})
        log = events.get_event_log(limit=1)
        assert len(log) == 1

    @patch("requests.post")
    def test_webhook_triggered(self, mock_post):
        events = EventSystem()
        events.add_webhook("https://example.com/hook", ["node_added"])
        events.emit("node_added", {"node_id": "n1"})
        time.sleep(0.1)
        mock_post.assert_called_once()
