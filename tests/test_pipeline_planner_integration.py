"""Integration tests for query planner and cost analyzer in retrieve_nodes_pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def _make_embedder(dim: int = 384):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestPlannerIntegration:
    def test_pipeline_with_planner_enabled(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, max_nodes=100, top_k=5,
            pipeline_enabled=True, pipeline_planner_enabled=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(content={"text": f"doc {i}"}, embedding=emb)
        result = mem.retrieve_nodes_pipeline("doc 1", top_k=5)
        assert "results" in result
        assert "metrics" in result
        # With no route set, planner should still work (standard plan)
        assert len(result["results"]) > 0

    def test_pipeline_with_cost_tracking(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, max_nodes=100, top_k=5,
            pipeline_enabled=True, pipeline_cost_tracking_enabled=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(content={"text": f"doc {i}"}, embedding=emb)
        result = mem.retrieve_nodes_pipeline("doc 1", top_k=5)
        assert "cost" in result
        assert result["cost"]["total_cost"] > 0
        assert "stage_costs" in result["cost"]
        assert "embed" in result["cost"]["stage_costs"]

    def test_pipeline_with_planner_and_cost(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, max_nodes=100, top_k=5,
            pipeline_enabled=True,
            pipeline_planner_enabled=True,
            pipeline_cost_tracking_enabled=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(content={"text": f"doc {i}"}, embedding=emb)
        result = mem.retrieve_nodes_pipeline("doc 1", top_k=5)
        assert "results" in result
        assert "cost" in result
        assert "metrics" in result

    def test_pipeline_plan_endpoint_preview(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, max_nodes=100, top_k=5,
            pipeline_enabled=True, pipeline_planner_enabled=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        pipeline = mem.build_pipeline()
        plan = pipeline.get_plan("test query", route="fast", top_k=5)
        assert "stage_names" in plan
        assert "estimated_latency_ms" in plan
        # Fast route should skip expensive stages
        assert "rerank" not in plan["stage_names"] or plan.get("skipped_stages")

    def test_planner_skips_stages_for_short_query(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, max_nodes=100, top_k=5,
            pipeline_enabled=True, pipeline_planner_enabled=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(content={"text": f"doc {i}"}, embedding=emb)
        result = mem.retrieve_nodes_pipeline("hi", top_k=3)
        assert "results" in result
        # Short query should skip explain
        stage_names = [m["stage"] for m in result["metrics"].get("stages", [])]
        # explain may be present as zero-latency skipped metric
        explain_metrics = [m for m in result["metrics"].get("stages", []) if m["stage"] == "explain"]
        if explain_metrics:
            assert explain_metrics[0]["latency_ms"] == 0.0
