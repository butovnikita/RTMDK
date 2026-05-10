"""Tests for webhook HTTP endpoints (/v1/webhooks)."""

import importlib

import pytest
from fastapi.testclient import TestClient

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def reset_webhook_manager():
    old = app_mod.webhook_manager
    app_mod.webhook_manager = None
    yield
    app_mod.webhook_manager = old


def _init_manager(tmp_path):
    from rtmdk.production.webhooks import WebhookManager

    mgr = WebhookManager(storage_path=str(tmp_path / "webhooks.json"))
    app_mod.webhook_manager = mgr
    return mgr


class TestWebhookSubscribe:
    def test_subscribe_success(self, client, tmp_path):
        _init_manager(tmp_path)
        resp = client.post(
            "/v1/webhooks",
            json={
                "url": "https://example.com/hook",
                "events": ["node_created", "node_deleted"],
                "secret": "shh",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "subscription_id" in data
        assert data["url"] == "https://example.com/hook"
        assert data["events"] == ["node_created", "node_deleted"]

    def test_subscribe_without_secret(self, client, tmp_path):
        _init_manager(tmp_path)
        resp = client.post(
            "/v1/webhooks",
            json={"url": "https://example.com/hook", "events": ["node_created"]},
        )
        assert resp.status_code == 200
        assert "subscription_id" in resp.json()

    def test_subscribe_missing_url(self, client, tmp_path):
        _init_manager(tmp_path)
        resp = client.post("/v1/webhooks", json={"events": ["node_created"]})
        assert resp.status_code == 422

    def test_subscribe_missing_events(self, client, tmp_path):
        _init_manager(tmp_path)
        resp = client.post("/v1/webhooks", json={"url": "https://example.com/hook"})
        assert resp.status_code == 422

    def test_subscribe_manager_not_available(self, client):
        # webhook_manager is None (reset_webhook_manager fixture)
        resp = client.post(
            "/v1/webhooks",
            json={"url": "https://example.com/hook", "events": ["node_created"]},
        )
        assert resp.status_code == 503


class TestWebhookUnsubscribe:
    def test_unsubscribe_success(self, client, tmp_path):
        mgr = _init_manager(tmp_path)
        sub = mgr.subscribe("https://example.com/hook", ["node_created"])
        resp = client.delete(f"/v1/webhooks/{sub.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unsubscribed"
        assert data["subscription_id"] == sub.id

    def test_unsubscribe_not_found(self, client, tmp_path):
        _init_manager(tmp_path)
        resp = client.delete("/v1/webhooks/nonexistent")
        assert resp.status_code == 404

    def test_unsubscribe_manager_not_available(self, client):
        resp = client.delete("/v1/webhooks/abc")
        assert resp.status_code == 503


class TestWebhookList:
    def test_list_empty(self, client, tmp_path):
        _init_manager(tmp_path)
        resp = client.get("/v1/webhooks")
        assert resp.status_code == 200
        assert resp.json()["subscriptions"] == []

    def test_list_with_subscriptions(self, client, tmp_path):
        mgr = _init_manager(tmp_path)
        mgr.subscribe("https://a.com/hook", ["node_created"])
        mgr.subscribe("https://b.com/hook", ["node_deleted"])
        resp = client.get("/v1/webhooks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["subscriptions"]) == 2
        urls = {s["url"] for s in data["subscriptions"]}
        assert urls == {"https://a.com/hook", "https://b.com/hook"}

    def test_list_manager_not_available(self, client):
        resp = client.get("/v1/webhooks")
        assert resp.status_code == 503
