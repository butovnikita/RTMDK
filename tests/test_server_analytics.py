"""Tests for RTMDK server analytics dashboard endpoints."""

import importlib

import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    # Disable API auth for tests
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def reset_analytics():
    """Reset analytics state between tests."""
    old_mem = app_mod.memory
    old_dash = app_mod.analytics_dashboard
    app_mod.memory = None
    app_mod.analytics_dashboard = None
    yield
    app_mod.memory = old_mem
    app_mod.analytics_dashboard = old_dash


def _make_memory():
    """Create a minimal RTMDKMemory with a field for testing."""
    import tempfile
    import numpy as np
    from rtmdk.memory.core import RTMDKField, RTMDKMemory
    from rtmdk.production.analytics_dashboard import AnalyticsDashboard
    from rtmdk.production.analytics_engine import AnalyticsEngine, AnalyticsStore
    from rtmdk.production.health_monitor import HealthMonitor

    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
    field = RTMDKField(cfg)
    field.add_node(
        embedding=np.array([0.0] * 16),
        content={"content": "hello world"},
        node_id="n0",
    )
    mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
    mem.field = field
    hm = HealthMonitor(memory=mem, check_interval=60)
    db_path = tempfile.mktemp(suffix=".db")
    store = AnalyticsStore(db_path=db_path)
    engine = AnalyticsEngine(store=store)
    dash = AnalyticsDashboard(mem, analytics_engine=engine, health_monitor=hm)
    return mem, dash


def test_analytics_overview_503_when_not_initialized(client):
    """Analytics overview returns 503 when dashboard not initialized."""
    resp = client.get("/v1/analytics/overview")
    assert resp.status_code == 503
    assert "not available" in resp.json()["detail"]


def test_analytics_memory_503_when_not_initialized(client):
    """Analytics memory returns 503 when dashboard not initialized."""
    resp = client.get("/v1/analytics/memory")
    assert resp.status_code == 503
    assert "not available" in resp.json()["detail"]


def test_analytics_events_503_when_not_initialized(client):
    """Analytics events returns 503 when dashboard not initialized."""
    resp = client.get("/v1/analytics/events")
    assert resp.status_code == 503
    assert "not available" in resp.json()["detail"]


def test_analytics_report_503_when_not_initialized(client):
    """Analytics report returns 503 when dashboard not initialized."""
    resp = client.get("/v1/analytics/report")
    assert resp.status_code == 503
    assert "not available" in resp.json()["detail"]


def test_analytics_track_503_when_not_initialized(client):
    """Analytics track returns 503 when dashboard not initialized."""
    resp = client.post("/v1/analytics/track", json={
        "event_type": "test",
        "properties": {"key": "value"},
    })
    assert resp.status_code == 503
    assert "not available" in resp.json()["detail"]


def test_analytics_overview_returns_data(client):
    """Analytics overview returns dashboard data."""
    mem, dash = _make_memory()
    app_mod.memory = mem
    app_mod.analytics_dashboard = dash
    try:
        resp = client.get("/v1/analytics/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "health" in data
        assert "queries" in data
        assert "version" in data
    finally:
        app_mod.memory = None
        app_mod.analytics_dashboard = None


def test_analytics_memory_returns_data(client):
    """Analytics memory returns memory-specific data."""
    mem, dash = _make_memory()
    app_mod.memory = mem
    app_mod.analytics_dashboard = dash
    try:
        resp = client.get("/v1/analytics/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert "topic_distribution" in data
        assert "forgetting_trends" in data
        assert "retrieval_stats" in data
        assert "node_lifecycle" in data
    finally:
        app_mod.memory = None
        app_mod.analytics_dashboard = None


def test_analytics_events_returns_data(client):
    """Analytics events returns event list."""
    mem, dash = _make_memory()
    app_mod.memory = mem
    app_mod.analytics_dashboard = dash
    try:
        dash.track_event("test_event", {"foo": "bar"}, session_id="s1")
        resp = client.get("/v1/analytics/events?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["event_type"] == "test_event"
    finally:
        app_mod.memory = None
        app_mod.analytics_dashboard = None


def test_analytics_events_filter_by_type(client):
    """Analytics events can be filtered by event type."""
    mem, dash = _make_memory()
    app_mod.memory = mem
    app_mod.analytics_dashboard = dash
    try:
        dash.track_event("type_a", {"k": 1})
        dash.track_event("type_b", {"k": 2})
        resp = client.get("/v1/analytics/events?event_type=type_a")
        assert resp.status_code == 200
        data = resp.json()
        assert all(e["event_type"] == "type_a" for e in data)
    finally:
        app_mod.memory = None
        app_mod.analytics_dashboard = None


def test_analytics_track_creates_event(client):
    """Analytics track creates a new event."""
    mem, dash = _make_memory()
    app_mod.memory = mem
    app_mod.analytics_dashboard = dash
    try:
        resp = client.post("/v1/analytics/track", json={
            "event_type": "user_action",
            "properties": {"action": "click"},
            "session_id": "sess-42",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        events = dash.get_event_series(event_type="user_action")
        assert len(events) == 1
        assert events[0]["properties"]["action"] == "click"
    finally:
        app_mod.memory = None
        app_mod.analytics_dashboard = None


def test_analytics_report_returns_combined_data(client):
    """Analytics report returns combined metrics."""
    mem, dash = _make_memory()
    app_mod.memory = mem
    app_mod.analytics_dashboard = dash
    try:
        resp = client.get("/v1/analytics/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "period" in data
        assert "event_counts" in data
        assert "time_series" in data
        assert "conversions" in data
    finally:
        app_mod.memory = None
        app_mod.analytics_dashboard = None
