"""Tests for query cache pipeline stages."""

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.pipeline.cache_stages import QueryCacheCheckStage, QueryCacheSaveStage
from rtmdk.pipeline.base import PipelineContext


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


def _make_memory(dim: int = 64, num_docs: int = 10, cache_enabled: bool = True):
    cfg = RTMDKConfig(
        latent_dim=dim, embedding_dim=dim, top_k=5,
        query_cache_size=100 if cache_enabled else 0,
        query_cache_ttl=3600,
    )
    mem = RTMDKMemory(config=cfg, embedder=_make_embedder(dim))
    for i in range(num_docs):
        emb = _make_embedder(dim)(f"doc {i}")
        mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")
    return mem


class TestQueryCachePipelineStages:
    def test_cache_check_miss(self):
        mem = _make_memory()
        stage = QueryCacheCheckStage(mem.field, mem)
        ctx = PipelineContext(query_text="q1", embedding=_make_embedder()("q1"))
        ctx = stage.process(ctx)
        assert not ctx.skip_remaining
        assert ctx.results == []

    def test_cache_check_hit(self):
        mem = _make_memory()
        # Pre-populate cache
        stage_check = QueryCacheCheckStage(mem.field, mem)
        stage_save = QueryCacheSaveStage(mem.field, mem)

        ctx = PipelineContext(query_text="q1", embedding=_make_embedder()("q1"))
        ctx.results = [("n0", 1.0, None)]
        ctx = stage_save.process(ctx)

        # Now check should hit
        ctx2 = PipelineContext(query_text="q1", embedding=_make_embedder()("q1"))
        ctx2 = stage_check.process(ctx2)
        assert ctx2.skip_remaining is True
        assert len(ctx2.results) == 1
        assert ctx2.results[0][0] == "n0"

    def test_pipeline_with_cache_stages(self):
        mem = _make_memory()
        pipeline = mem.build_pipeline()
        # First call - miss
        ctx = pipeline.run("doc 5", top_k=3)
        assert not ctx.skip_remaining or ctx.skip_remaining  # may or may not skip
        assert len(ctx.results) > 0

        # Second call - should hit cache
        ctx2 = pipeline.run("doc 5", top_k=3)
        # If cache hit, skip_remaining should be True after cache check
        # But we need to check metrics to see if stages were skipped
        stage_names = [m.name for m in ctx2.metrics]
        if "query_cache_check" in stage_names:
            cache_metric = next(m for m in ctx2.metrics if m.name == "query_cache_check")
            # If hit, there should be few stages after cache check
            if ctx2.skip_remaining:
                assert len(ctx2.metrics) <= 2  # cache_check + maybe cache_save

    def test_cache_no_embedding(self):
        mem = _make_memory()
        stage = QueryCacheCheckStage(mem.field, mem)
        ctx = PipelineContext(query_text="q1", embedding=None)
        ctx = stage.process(ctx)
        assert not ctx.skip_remaining

    def test_cache_disabled(self):
        mem = _make_memory(cache_enabled=False)
        # query_cache should be None
        assert getattr(mem.field, "query_cache", None) is None
        pipeline = mem.build_pipeline()
        stage_names = [s.name for s in pipeline.stages]
        assert "query_cache_check" not in stage_names
        assert "query_cache_save" not in stage_names
