"""Tests for rtmdk.production.analytics."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.production.analytics import MemoryAnalytics


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestMemoryAnalytics:
    def test_topic_distribution(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        analytics = MemoryAnalytics(mem)
        dist = analytics.get_topic_distribution()
        # Default tier is semantic unless changed
        assert "semantic" in dist or "unknown" in dist
        assert sum(dist.values()) == 1

    def test_forgetting_trends(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        analytics = MemoryAnalytics(mem)
        trends = analytics.get_forgetting_trends()
        assert len(trends) == 4
        categories = {t["category"] for t in trends}
        assert categories == {"high", "medium", "low", "critical"}

    def test_retrieval_stats(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        analytics = MemoryAnalytics(mem)
        stats = analytics.get_retrieval_stats()
        assert "total_queries" in stats
        assert "bm25_fallbacks" in stats

    def test_node_lifecycle_empty(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        analytics = MemoryAnalytics(mem)
        lifecycle = analytics.get_node_lifecycle()
        assert lifecycle["count"] == 0

    def test_node_lifecycle_with_nodes(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "hello", "session_id": "s1"}, {"output": "world"})
        analytics = MemoryAnalytics(mem)
        lifecycle = analytics.get_node_lifecycle()
        assert lifecycle["count"] == 1
        assert lifecycle["avg_age_hours"] >= 0

    def test_export_report(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        analytics = MemoryAnalytics(mem)
        report = analytics.export_report()
        assert "topic_distribution" in report
        assert "forgetting_trends" in report
        assert "retrieval_stats" in report
        assert "node_lifecycle" in report
        assert "timestamp" in report
