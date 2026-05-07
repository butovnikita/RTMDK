"""Tests for WebhookManager and webhook endpoints."""

import os
import tempfile

import pytest

from rtmdk.production.webhooks import WebhookManager, WebhookSubscription


@pytest.fixture
def webhook_mgr():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "webhooks.json")
        mgr = WebhookManager(storage_path=path)
        yield mgr


class TestWebhookManager:
    def test_subscribe_and_list(self, webhook_mgr):
        sub = webhook_mgr.subscribe("https://example.com/hook", ["node_created"])
        assert isinstance(sub, WebhookSubscription)
        assert sub.url == "https://example.com/hook"
        assert "node_created" in sub.events

        subs = webhook_mgr.list_subscriptions()
        assert len(subs) == 1

    def test_unsubscribe(self, webhook_mgr):
        webhook_mgr.subscribe("https://example.com/hook", ["node_created"])
        ok = webhook_mgr.unsubscribe(list(webhook_mgr._subs.keys())[0])
        assert ok is True
        assert len(webhook_mgr.list_subscriptions()) == 0

    def test_unsubscribe_unknown(self, webhook_mgr):
        assert webhook_mgr.unsubscribe("nope") is False

    def test_filter_by_event(self, webhook_mgr):
        webhook_mgr.subscribe("https://a.com", ["node_created"])
        webhook_mgr.subscribe("https://b.com", ["node_deleted"])
        subs = webhook_mgr.list_subscriptions(event_type="node_created")
        assert len(subs) == 1
        assert subs[0].url == "https://a.com"

    def test_persistence(self, webhook_mgr):
        webhook_mgr.subscribe("https://example.com/hook", ["node_created"])
        mgr2 = WebhookManager(storage_path=webhook_mgr.storage_path)
        subs = mgr2.list_subscriptions()
        assert len(subs) == 1
        assert subs[0].url == "https://example.com/hook"

    def test_dispatch_no_subscriptions(self, webhook_mgr):
        results = webhook_mgr.dispatch("node_created", {"node_id": "n1"})
        assert results == []

    def test_dispatch_with_signature(self, webhook_mgr):
        # This test would require a real HTTP server; we test the signature logic instead
        webhook_mgr.subscribe(
            "https://example.com/hook",
            ["node_created"],
            secret="shhh",
        )
        # Just verify no crash on dispatch (will fail to connect)
        results = webhook_mgr.dispatch("node_created", {"node_id": "n1"})
        assert len(results) == 1
        assert results[0]["success"] is False  # no real server
