"""Unit tests for PipelineBuilder."""

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.pipeline_builder import PipelineBuilder
from rtmdk.pipeline import PipelineExecutor, PlannedPipelineExecutor


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestPipelineBuilder:
    def test_build_returns_executor(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, pipeline_breaker_enabled=False)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        builder = PipelineBuilder(mem)
        executor = builder.build()
        assert isinstance(executor, PipelineExecutor)

    def test_build_contains_expected_stages(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, pipeline_breaker_enabled=False)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        builder = PipelineBuilder(mem)
        executor = builder.build()

        stage_names = [s.name for s in executor.stages]
        assert "embed" in stage_names
        assert "route" in stage_names
        assert "retrieve" in stage_names
        assert "rerank" in stage_names
        assert "calibrate" in stage_names
        assert "explain" in stage_names

    def test_build_with_planner_returns_planned_executor(self):
        cfg = RTMDKConfig(
            latent_dim=64, embedding_dim=64,
            pipeline_breaker_enabled=False,
            pipeline_planner_enabled=True,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        builder = PipelineBuilder(mem)
        executor = builder.build()
        assert isinstance(executor, PlannedPipelineExecutor)

    def test_build_with_breaker_attaches_breaker(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, pipeline_breaker_enabled=True)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        builder = PipelineBuilder(mem)
        executor = builder.build()

        for stage in executor.stages:
            assert stage.circuit_breaker is not None
            assert stage.circuit_breaker.name == stage.name

    def test_build_without_breaker_no_breaker(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, pipeline_breaker_enabled=False)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        builder = PipelineBuilder(mem)
        executor = builder.build()

        for stage in executor.stages:
            assert stage.circuit_breaker is None
