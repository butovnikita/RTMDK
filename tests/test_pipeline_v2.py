"""Tests for rtmdk.pipeline v2 (batch + registry)."""

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.pipeline.base import PipelineContext, PipelineStage
from rtmdk.pipeline.stages import EmbedStage, RetrieveStage
from rtmdk.pipeline.executor import PipelineExecutor
from rtmdk.pipeline.batch import BatchEmbedStage, BatchPipelineExecutor
from rtmdk.pipeline.registry import StageRegistry, GLOBAL_REGISTRY


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


def _make_memory(dim: int = 64, num_docs: int = 10):
    cfg = RTMDKConfig(latent_dim=dim, embedding_dim=dim, top_k=5)
    mem = RTMDKMemory(config=cfg, embedder=_make_embedder(dim))
    for i in range(num_docs):
        emb = _make_embedder(dim)(f"doc {i}")
        mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")
    return mem


class TestBatchPipeline:
    def test_batch_embed_stage_single(self):
        mem = _make_memory()
        stage = BatchEmbedStage(mem.embedder)
        ctx = PipelineContext(query_text="q1")
        ctx = stage.run(ctx)
        assert ctx.embedding is not None
        assert isinstance(ctx.embedding, np.ndarray)

    def test_batch_pipeline_executor(self):
        mem = _make_memory()
        batch = BatchPipelineExecutor([
            BatchEmbedStage(mem.embedder),
            RetrieveStage(mem.field),
        ])
        outputs = batch.run_batch(["q1", "q2"], top_k=3)
        assert len(outputs) == 2
        assert all(o["results_count"] <= 3 for o in outputs)
        assert all(o["total_latency_ms"] > 0 for o in outputs)
        assert all(len(o["stages"]) > 0 for o in outputs)

    def test_batch_metrics(self):
        mem = _make_memory()
        batch = BatchPipelineExecutor([
            BatchEmbedStage(mem.embedder),
            RetrieveStage(mem.field),
        ])
        outputs = batch.run_batch(["q"], top_k=3)
        assert "stages" in outputs[0]
        assert isinstance(outputs[0]["stages"], list)
        assert outputs[0]["total_latency_ms"] > 0


class TestStageRegistry:
    def test_register_and_create(self):
        class MyStage(PipelineStage):
            name = "my_stage"
            def process(self, ctx):
                ctx.results = [("x", 1.0, None)]
                return ctx
        reg = StageRegistry()
        reg.register("my_stage", MyStage)
        stage = reg.create("my_stage")
        assert isinstance(stage, MyStage)
        ctx = stage.run(PipelineContext(query_text="q"))
        assert len(ctx.results) == 1

    def test_create_missing(self):
        reg = StageRegistry()
        with pytest.raises(KeyError):
            reg.create("missing")

    def test_list_stages(self):
        reg = StageRegistry()
        reg.register("a", PipelineStage)
        assert "a" in reg.list_stages()

    def test_global_registry(self):
        # GLOBAL_REGISTRY should have the default stages registered
        assert "embed" in GLOBAL_REGISTRY.list_stages()
        assert "retrieve" in GLOBAL_REGISTRY.list_stages()
        stage = GLOBAL_REGISTRY.create("embed", embedder=lambda x: x)  # dummy embedder
        assert isinstance(stage, EmbedStage)

    def test_global_register_duplicate(self):
        with pytest.raises(ValueError):
            GLOBAL_REGISTRY.register("embed", EmbedStage)


class TestPipelineStageRegistryIntegration:
    def test_build_pipeline_from_registry(self):
        mem = _make_memory()
        stages = [
            GLOBAL_REGISTRY.create("embed", embedder=mem.embedder),
            GLOBAL_REGISTRY.create("retrieve", field=mem.field),
        ]
        pipe = PipelineExecutor(stages)
        ctx = pipe.run(query_text="q1", top_k=3)
        assert len(ctx.results) <= 3
