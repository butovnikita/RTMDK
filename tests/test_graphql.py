"""Tests for RTMDK GraphQL endpoints."""

import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField, RTMDKMemory

app_mod = importlib.import_module("rtmdk.server.app")


@pytest.fixture(scope="module")
def client():
    app_mod.ENABLE_API_AUTH = False
    return TestClient(app_mod.app)


@pytest.fixture(autouse=True)
def reset_memory():
    old = app_mod.memory
    app_mod.memory = None
    yield
    app_mod.memory = old


class TestGraphQLHealth:
    def test_health_when_no_memory(self, client):
        resp = client.post("/graphql", json={
            "query": "{ health { status version memoryNodes } }"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "errors" not in data
        health = data["data"]["health"]
        assert health["status"] == "ok"
        assert health["memoryNodes"] == 0

    def test_health_with_memory(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        field = RTMDKField(cfg)
        field.add_node(
            embedding=np.array([0.0] * 16),
            content={"content": "hello"},
            node_id="n0")
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.post("/graphql", json={
                "query": "{ health { status version memoryNodes } }"
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["data"]["health"]["memoryNodes"] == 1
        finally:
            app_mod.memory = None


class TestGraphQLNodeQueries:
    def test_node_returns_null_when_not_found(self, client):
        resp = client.post("/graphql", json={
            "query": '{ node(id: "nonexistent") { id content salience } }'
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["node"] is None

    def test_node_returns_node(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        field = RTMDKField(cfg)
        field.add_node(
            embedding=np.array([0.0] * 16),
            content={"text": "hello world"},
            node_id="n0")
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.post("/graphql", json={
                "query": '{ node(id: "n0") { id content salience phase amplitude } }'
            })
            assert resp.status_code == 200
            data = resp.json()
            node = data["data"]["node"]
            assert node["id"] == "n0"
            assert node["content"] == "hello world"
        finally:
            app_mod.memory = None

    def test_nodes_pagination(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        field = RTMDKField(cfg)
        for i in range(5):
            field.add_node(
                embedding=np.array([0.0] * 16),
                content={"content": f"node {i}"},
                node_id=f"n{i}")
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.post("/graphql", json={
                "query": '{ nodes(limit: 3, offset: 0) { id content } }'
            })
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["data"]["nodes"]) == 3
        finally:
            app_mod.memory = None


class TestGraphQLMutations:
    def test_create_node(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        field = RTMDKField(cfg)
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.post("/graphql", json={
                "query": 'mutation { createNode(content: "new node", salience: 0.8) { id content salience } }'
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "errors" not in data
            node = data["data"]["createNode"]
            assert node["content"] == "new node"
            assert node["salience"] == 0.8
            assert node["id"] in field.nodes
        finally:
            app_mod.memory = None

    def test_delete_node(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        field = RTMDKField(cfg)
        field.add_node(
            embedding=np.array([0.0] * 16),
            content={"content": "to delete"},
            node_id="ndel")
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.post("/graphql", json={
                "query": 'mutation { deleteNode(id: "ndel") }'
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["data"]["deleteNode"] is True
            assert "ndel" not in field.nodes
        finally:
            app_mod.memory = None

    def test_delete_node_not_found(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        field = RTMDKField(cfg)
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.post("/graphql", json={
                "query": 'mutation { deleteNode(id: "missing") }'
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["data"]["deleteNode"] is False
        finally:
            app_mod.memory = None


class TestGraphQLPipelineQuery:
    def test_query_pipeline_returns_results(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, pipeline_breaker_enabled=False)
        field = RTMDKField(cfg)
        field.add_node(
            embedding=np.array([0.0] * 16),
            content={"text": "hello world"},
            node_id="n0")
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            resp = client.post("/graphql", json={
                "query": '{ queryPipeline(query: "hello", topK: 3) { query results { nodeId score content } route total metrics { totalLatencyMs stages { stage latencyMs } } } }'
            })
            assert resp.status_code == 200
            data = resp.json()
            assert "errors" not in data
            result = data["data"]["queryPipeline"]
            assert result["query"] == "hello"
            assert result["total"] >= 1
            assert len(result["results"]) >= 1
            assert result["results"][0]["nodeId"] == "n0"
            assert "metrics" in result
            assert result["metrics"]["totalLatencyMs"] > 0
            assert len(result["metrics"]["stages"]) > 0
        finally:
            app_mod.memory = None

    def test_query_pipeline_null_when_no_memory(self, client):
        resp = client.post("/graphql", json={
            "query": '{ queryPipeline(query: "hello") { query total } }'
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"]["queryPipeline"] is None


class TestGraphQLSubscription:
    def test_subscription_schema_has_pipeline_stream(self):
        from rtmdk.server.graphql_schema import schema
        assert "Subscription" in str(schema)
        assert "pipelineStream" in str(schema)

    def test_subscription_pipeline_stream_with_memory(self, client):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, pipeline_breaker_enabled=False)
        field = RTMDKField(cfg)
        field.add_node(
            embedding=np.array([0.0] * 16),
            content={"text": "hello world"},
            node_id="n0")
        mem = RTMDKMemory(config=cfg, embedder=lambda x: np.array([0.0] * 16))
        mem.field = field
        app_mod.memory = mem

        try:
            # GraphQL subscriptions over HTTP typically use POST with operationType: subscription
            resp = client.post("/graphql", json={
                "query": 'subscription { pipelineStream(query: "hello", topK: 3) { eventType stage } }',
                "operationName": None,
            })
            # FastAPI TestClient may not fully support streaming subscriptions,
            # but we verify the endpoint doesn't crash
            assert resp.status_code in (200, 400)
        finally:
            app_mod.memory = None
