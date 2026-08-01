"""Tests for RTMDK MCP Server (Track 8)."""

import json
import numpy as np
import pytest
from rtmdk.mcp_server import _ctx, add_memory, query_memory, delete_memory, consolidate_memory, get_memory_stats
from rtmdk.mcp_server import memory_stats, memory_nodes, memory_node, memory_context_prompt
from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


@pytest.fixture(autouse=True)
def setup_memory(tmp_path):
    """Initialize fresh RTMDKMemory for each MCP test."""
    cfg = RTMDKConfig(
        latent_dim=64,
        use_hnsw=False,
        hyperbolic=False,
        quantization="none",
        enable_engrams=False,
    )
    embedder = _make_embedder(64)
    _ctx.memory = RTMDKMemory(config=cfg, embedder=embedder, wal_path=None)
    yield
    _ctx.memory = None


class TestMCPTools:
    def test_add_memory(self):
        result = add_memory("hello world", session_id="s1", modality="text")
        assert "Memory added" in result
        assert len(_ctx.memory.field.nodes) == 1

    def test_query_memory(self):
        add_memory("coffee is great", session_id="s1")
        result = query_memory("coffee", top_k=3, session_id="s1")
        assert "coffee" in result.lower()

    def test_delete_memory(self):
        add_memory("delete me", session_id="s1")
        nid = list(_ctx.memory.field.nodes.keys())[0]
        result = delete_memory(nid)
        assert "Deleted" in result
        assert len(_ctx.memory.field.nodes) == 0

    def test_delete_memory_not_found(self):
        result = delete_memory("nonexistent")
        assert "not found" in result

    def test_consolidate_memory(self):
        add_memory("node 1", session_id="s1")
        add_memory("node 2", session_id="s1")
        result = consolidate_memory()
        assert "Consolidation complete" in result

    def test_get_memory_stats(self):
        add_memory("stat test", session_id="s1")
        result = get_memory_stats()
        stats = json.loads(result)
        assert stats["active_nodes"] == 1
        assert "total_adds" in stats


class TestMCPResources:
    def test_memory_stats_resource(self):
        add_memory("resource test", session_id="s1")
        result = memory_stats()
        stats = json.loads(result)
        assert stats["active_nodes"] == 1

    def test_memory_nodes_resource(self):
        add_memory("node list", session_id="s1")
        result = memory_nodes()
        nids = json.loads(result)
        assert len(nids) == 1

    def test_memory_node_resource(self):
        add_memory("single node", session_id="s1")
        nid = list(_ctx.memory.field.nodes.keys())[0]
        result = memory_node(nid)
        data = json.loads(result)
        assert data["id"] == nid

    def test_memory_node_resource_not_found(self):
        result = memory_node("missing")
        assert "error" in json.loads(result)


class TestMCPPrompts:
    def test_memory_context_prompt(self):
        add_memory("context about coffee", session_id="s1")
        result = memory_context_prompt("coffee")
        assert "coffee" in result.lower()
        assert "long-term memory" in result.lower()

    def test_memory_context_prompt_empty(self):
        result = memory_context_prompt("")
        assert "helpful assistant" in result.lower()
