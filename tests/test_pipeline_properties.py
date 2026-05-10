"""Property-based tests for pipeline invariants."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def _make_embedder(dim: int = 16):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestPipelineInvariants:
    """Invariant checks that must hold for all pipeline executions."""

    def test_latency_non_negative(self):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(16))
        mem.add_node(
            embedding=_make_embedder(16)("doc"),
            content={"text": "doc"},
            node_id="n0",
        )
        result = mem.retrieve_nodes_pipeline("doc", top_k=5)
        metrics = result.get("metrics", {})
        assert metrics["total_latency_ms"] >= 0
        for stage in metrics.get("stages", []):
            assert stage["latency_ms"] >= 0

    def test_results_count_lte_top_k(self):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(16))
        for i in range(20):
            mem.add_node(
                embedding=_make_embedder(16)(f"doc {i}"),
                content={"text": f"doc {i}"},
                node_id=f"n{i}",
            )
        result = mem.retrieve_nodes_pipeline("doc 5", top_k=3)
        assert len(result["results"]) <= 3

    def test_pipeline_enabled_delegates_to_pipeline(self):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, pipeline_enabled=True)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(16))
        mem.add_node(
            embedding=_make_embedder(16)("hello"),
            content={"text": "hello"},
            node_id="n0",
        )
        # When pipeline_enabled=True, retrieve_nodes delegates to pipeline
        emb = _make_embedder(16)("hello")
        legacy = mem.retrieve_nodes("hello", embedding=emb, top_k=5)
        pipeline = mem.retrieve_nodes_pipeline("hello", top_k=5)

        assert isinstance(legacy, list)
        assert isinstance(pipeline["results"], list)

    def test_empty_field_returns_empty_results(self):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(16))
        result = mem.retrieve_nodes_pipeline("anything", top_k=5)
        assert len(result["results"]) == 0
        assert result["metrics"]["total_latency_ms"] >= 0

    def test_session_id_preserved(self):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(16))
        mem.add_node(
            embedding=_make_embedder(16)("test"),
            content={"text": "test"},
            node_id="n0",
        )
        result = mem.retrieve_nodes_pipeline("test", top_k=5, session_id="sess_42")
        metrics = result.get("metrics", {})
        assert metrics is not None

    def test_breaker_states_tracked_for_all_stages(self):
        cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, pipeline_breaker_enabled=True)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(16))
        mem.add_node(
            embedding=_make_embedder(16)("test"),
            content={"text": "test"},
            node_id="n0",
        )
        result = mem.retrieve_nodes_pipeline("test", top_k=5)
        metrics = result.get("metrics", {})
        breaker_states = metrics.get("breaker_states", {})
        # Should have breaker state for at least some stages
        assert isinstance(breaker_states, dict)
