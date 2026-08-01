"""Smoke tests for rtmdk.production.integration."""

import numpy as np
from rtmdk.production.integration import ProductionRTMDK, ProductionConfig


def _embed(text: str) -> np.ndarray:
    return np.random.randn(768).astype(np.float32)


class TestProductionRTMDK:
    def test_init_default(self):
        prod = ProductionRTMDK(embedder=_embed)
        assert prod.memory is not None
        assert prod.production_mode is True

    def test_save_and_load(self):
        prod = ProductionRTMDK(embedder=_embed)
        prod.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        result = prod.load_memory_variables({"input": "hello", "session_id": "s1"})
        assert "rtmdk_context" in result

    def test_load_memory_variables_empty_query(self):
        prod = ProductionRTMDK(embedder=_embed)
        result = prod.load_memory_variables({})
        assert result == {"rtmdk_context": ""}

    def test_apply_feedback(self):
        prod = ProductionRTMDK(embedder=_embed)
        prod.apply_feedback("q", 0.8)

    def test_prune_memory(self):
        prod = ProductionRTMDK(embedder=_embed)
        n = prod.prune_memory()
        assert n >= 0

    def test_get_production_stats(self):
        prod = ProductionRTMDK(embedder=_embed)
        stats = prod.get_production_stats()
        assert "query_count" in stats
        assert "uptime_seconds" in stats

    def test_custom_config(self):
        cfg = ProductionConfig()
        cfg.top_k = 3
        prod = ProductionRTMDK(embedder=_embed, config=cfg)
        assert prod.pconfig.top_k == 3
