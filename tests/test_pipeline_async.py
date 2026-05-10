"""Tests for async pipeline execution."""

import asyncio

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.pipeline.executor import PipelineExecutor
from rtmdk.pipeline.stages import EmbedStage, RetrieveStage


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestPipelineExecutorAsync:
    def test_run_async(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        pipeline = mem.build_pipeline()
        ctx = asyncio.run(pipeline.run_async("doc 5", top_k=3))
        assert len(ctx.results) > 0
        assert ctx.query_text == "doc 5"

    def test_run_batch_async(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        pipeline = mem.build_pipeline()
        ctxs = asyncio.run(pipeline.run_batch_async(["doc 2", "doc 5", "doc 8"], top_k=3))
        assert len(ctxs) == 3
        for ctx in ctxs:
            assert len(ctx.results) > 0

    def test_retrieve_nodes_pipeline_async(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64, top_k=5,
            pipeline_enabled=True,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        result = asyncio.run(mem.retrieve_nodes_pipeline_async("doc 5", top_k=3))
        assert "results" in result
        assert "metrics" in result
        assert len(result["results"]) > 0

    def test_async_does_not_block(self):
        """Verify that run_async truly runs in a thread pool."""
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, top_k=5)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        pipeline = mem.build_pipeline()

        async def concurrent_queries():
            tasks = [
                pipeline.run_async(f"doc {i}", top_k=3)
                for i in range(5)
            ]
            return await asyncio.gather(*tasks)

        ctxs = asyncio.run(concurrent_queries())
        assert len(ctxs) == 5
        for ctx in ctxs:
            assert len(ctx.results) > 0
