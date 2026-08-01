"""Tests for pipeline memory profiler."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.pipeline.profiler import PipelineMemoryProfiler
from rtmdk.pipeline.stages import EmbedStage
from rtmdk.pipeline.base import PipelineContext


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class TestPipelineMemoryProfiler:
    def test_profiler_tracks_memory(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        pipeline = mem.build_pipeline()
        ctx, profiler = pipeline.run_with_profiler("doc 5", top_k=3)

        assert len(ctx.results) > 0
        summary = profiler.get_summary()
        assert len(summary) > 0
        # Each stage should have memory data
        for stage_name, stats in summary.items():
            assert stats["runs"] >= 1
            assert stats["peak_bytes_max"] >= 0

    def test_profiler_resets(self):
        profiler = PipelineMemoryProfiler()
        stage = EmbedStage(_make_embedder(64))
        ctx = PipelineContext(query_text="q")
        profiler.profile_stage(stage, ctx)
        assert len(profiler.get_summary()) > 0
        profiler.reset()
        assert len(profiler.get_summary()) == 0

    def test_profiler_start_stop(self):
        import tracemalloc

        if tracemalloc.is_tracing():
            tracemalloc.stop()
        profiler = PipelineMemoryProfiler()
        profiler.start()
        assert profiler._enabled is True
        profiler.stop()
        assert profiler._enabled is False

    def test_profiler_attaches_to_context(self):
        profiler = PipelineMemoryProfiler()
        stage = EmbedStage(_make_embedder(64))
        ctx = PipelineContext(query_text="q")
        ctx = profiler.profile_stage(stage, ctx)
        assert hasattr(ctx, "memory_usage")
        assert "embed" in ctx.memory_usage  # type: ignore
        assert "peak_bytes" in ctx.memory_usage["embed"]  # type: ignore
