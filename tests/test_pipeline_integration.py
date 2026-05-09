"""Integration tests for retrieve_nodes_pipeline() end-to-end."""

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.pipeline import BatchPipelineExecutor


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestRetrieveNodesPipeline:
    def test_pipeline_returns_results(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(20):
            emb = _make_embedder(64)(f"document about topic {i}")
            mem.add_node(embedding=emb, content={"text": f"document about topic {i}"}, node_id=f"n{i}")

        result = mem.retrieve_nodes_pipeline("document about topic 5", top_k=3)
        assert "results" in result
        assert "route" in result
        assert "explanations" in result
        assert "metrics" in result
        assert len(result["results"]) <= 3
        assert len(result["results"]) > 0
        # Results should be tuples: (node_id, score, node)
        assert all(len(r) == 3 for r in result["results"])

    def test_pipeline_metrics_structure(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        result = mem.retrieve_nodes_pipeline("doc 3", top_k=3)
        metrics = result["metrics"]
        assert "stages" in metrics
        assert "total_latency_ms" in metrics
        assert isinstance(metrics["stages"], list)
        assert len(metrics["stages"]) > 0
        for stage_metric in metrics["stages"]:
            assert "stage" in stage_metric
            assert "latency_ms" in stage_metric
            assert "error" in stage_metric
            assert "degraded" in stage_metric

    def test_pipeline_breaker_states(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=5,
            pipeline_breaker_enabled=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        result = mem.retrieve_nodes_pipeline("doc 3", top_k=3)
        metrics = result["metrics"]
        assert "breaker_states" in metrics
        # All breakers should be closed under normal conditions
        for state in metrics["breaker_states"].values():
            assert state == "closed"

    def test_pipeline_with_session_id(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"session test {i}")
            mem.add_node(embedding=emb, content={"text": f"session test {i}"}, node_id=f"n{i}")

        result = mem.retrieve_nodes_pipeline("session test 5", top_k=3, session_id="sess_1")
        assert len(result["results"]) > 0

    def test_pipeline_with_provided_embedding(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"embedding test {i}")
            mem.add_node(embedding=emb, content={"text": f"embedding test {i}"}, node_id=f"n{i}")

        query_emb = _make_embedder(64)("embedding test 5")
        result = mem.retrieve_nodes_pipeline("ignored", embedding=query_emb, top_k=3)
        assert len(result["results"]) > 0

    def test_pipeline_empty_field(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        result = mem.retrieve_nodes_pipeline("anything", top_k=3)
        assert result["results"] == []
        assert result["metrics"]["results_count"] == 0


class TestBatchPipelineIntegration:
    def test_batch_pipeline_executor(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(20):
            emb = _make_embedder(64)(f"batch doc {i}")
            mem.add_node(embedding=emb, content={"text": f"batch doc {i}"}, node_id=f"n{i}")

        batch = BatchPipelineExecutor(mem.build_pipeline().stages)
        outputs = batch.run_batch(["batch doc 5", "batch doc 10", "batch doc 15"], top_k=3)
        assert len(outputs) == 3
        for output in outputs:
            assert output["results_count"] > 0
            assert output["total_latency_ms"] > 0
            assert len(output["stages"]) > 0

    def test_batch_metrics_per_query(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"batch metric {i}")
            mem.add_node(embedding=emb, content={"text": f"batch metric {i}"}, node_id=f"n{i}")

        batch = BatchPipelineExecutor(mem.build_pipeline().stages)
        outputs = batch.run_batch(["batch metric 2", "batch metric 7"], top_k=2)
        assert len(outputs) == 2
        for output in outputs:
            assert "stages" in output
            assert "total_latency_ms" in output
            assert "breaker_states" in output
