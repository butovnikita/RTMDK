"""Tests for pipeline A/B testing framework."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.pipeline.ab_testing import PipelineABTester


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestPipelineABTester:
    def test_compare_single(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=5,
            pipeline_enabled=True,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        tester = PipelineABTester(mem)
        result = tester.compare_single("doc 5", top_k=3)

        assert "legacy" in result
        assert "pipeline" in result
        assert "comparison" in result
        assert len(result["legacy"]["results"]) > 0
        assert len(result["pipeline"]["results"]) > 0
        assert result["legacy"]["latency_ms"] > 0
        assert result["pipeline"]["latency_ms"] > 0
        assert 0.0 <= result["comparison"]["jaccard_overlap"] <= 1.0
        assert 0.0 <= result["comparison"]["kendall_tau"] <= 1.0

    def test_compare_batch(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=5,
            pipeline_enabled=True,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        tester = PipelineABTester(mem)
        results = tester.compare_batch(["doc 2", "doc 5", "doc 8"], top_k=3)

        assert len(results) == 3
        for r in results:
            assert len(r["legacy"]["results"]) > 0
            assert len(r["pipeline"]["results"]) > 0

    def test_summary(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=5,
            pipeline_enabled=True,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        tester = PipelineABTester(mem)
        tester.compare_batch(["doc 1", "doc 3", "doc 7"], top_k=3)

        summary = tester.summary()
        assert summary["runs"] == 3
        assert "legacy_latency_ms" in summary
        assert "pipeline_latency_ms" in summary
        assert "jaccard_overlap" in summary
        assert "kendall_tau" in summary
        assert summary["legacy_latency_ms"]["mean"] > 0
        assert summary["pipeline_latency_ms"]["mean"] > 0

    def test_empty_summary(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        tester = PipelineABTester(mem)
        summary = tester.summary()
        assert summary["runs"] == 0

    def test_compare_with_embedding(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=5,
            pipeline_enabled=True,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        tester = PipelineABTester(mem)
        query_emb = _make_embedder(64)("doc 5")
        result = tester.compare_single("doc 5", top_k=3, embedding=query_emb)
        assert len(result["legacy"]["results"]) > 0
        assert len(result["pipeline"]["results"]) > 0
