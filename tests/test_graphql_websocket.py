"""Runtime WebSocket tests for GraphQL subscriptions."""

import importlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField, RTMDKMemory

app_mod = importlib.import_module("rtmdk.server.app")

GRAPHQL_AVAILABLE = getattr(app_mod, "GRAPHQL_AVAILABLE", False)


def _embedder(text: str):
    return np.array([0.0] * 16)


def _init_memory():
    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, pipeline_breaker_enabled=False)
    field = RTMDKField(cfg)
    field.add_node(
        embedding=np.array([0.0] * 16),
        content={"text": "hello world"},
        node_id="n0",
    )
    mem = RTMDKMemory(config=cfg, embedder=_embedder)
    mem.field = field
    app_mod.memory = mem
    return mem


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


@pytest.mark.skipif(not GRAPHQL_AVAILABLE, reason="graphql not available")
class TestGraphQLWebSocketSubscription:
    def test_websocket_subscription_pipeline_stream(self, client):
        _init_memory()
        with client.websocket_connect("/graphql", subprotocols=["graphql-transport-ws"]) as ws:
            # 1. Initialize connection
            ws.send_json({"type": "connection_init"})
            msg = ws.receive_json()
            assert msg["type"] == "connection_ack"

            # 2. Subscribe to pipeline stream
            ws.send_json(
                {
                    "type": "subscribe",
                    "id": "sub-1",
                    "payload": {
                        "query": ('subscription { pipelineStream(query: "hello", topK: 3) ' "{ eventType stage } }"),
                    },
                }
            )

            # 3. Receive at least one event (next) or completion
            responses = []
            for _ in range(10):
                msg = ws.receive_json()
                responses.append(msg)
                if msg["type"] in ("next", "complete", "error"):
                    break

            # Should receive either data or complete
            assert any(r["type"] in ("next", "complete") for r in responses)

            # 4. Unsubscribe / close
            ws.send_json({"type": "complete", "id": "sub-1"})

    def test_websocket_subscription_without_memory_returns_error(self, client):
        # memory is None (reset_memory fixture)
        with client.websocket_connect("/graphql", subprotocols=["graphql-transport-ws"]) as ws:
            ws.send_json({"type": "connection_init"})
            msg = ws.receive_json()
            assert msg["type"] == "connection_ack"

            ws.send_json(
                {
                    "type": "subscribe",
                    "id": "sub-2",
                    "payload": {
                        "query": ('subscription { pipelineStream(query: "hello", topK: 3) ' "{ eventType stage } }"),
                    },
                }
            )

            # Should receive error because memory not initialized
            msg = ws.receive_json()
            # Strawberry wraps execution errors in 'next' with errors payload
            assert msg["type"] == "next"
            assert "sub-2" == msg.get("id", "")
            assert "errors" in msg.get("payload", {})
