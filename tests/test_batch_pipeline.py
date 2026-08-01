"""Tests for batch pipeline execution."""

from __future__ import annotations

import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory
from rtmdk.pipeline.batch import BatchPipelineExecutor


@pytest.fixture
def memory():
    cfg = RTMDKConfig(
        latent_dim=64,
        embedding_dim=64,
        max_nodes=500,
        top_k=5,
        min_response=0.0,
        bandwidth=10.0,
        phase_coupling=0.0,
        use_hnsw=False,
        bm25_fallback=False,
    )

    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        r = np.random.default_rng(h)
        return r.standard_normal(64).astype(np.float32)

    mem = RTMDKMemory(config=cfg, embedder=embed)
    for i in range(100):
        text = f"document about topic {i % 10} with keywords neural search vector"
        mem.add_node(content={"text": text}, embedding=embed(text))
    return mem


def test_query_batch_returns_results(memory: RTMDKMemory):
    """query_batch should return results for multiple queries."""
    queries = ["document about topic 0", "document about topic 5", "document about topic 9"]
    embeddings = np.array([memory.embedder(q) for q in queries])
    results = memory.field.query_batch(embeddings, top_k=5)
    assert len(results) == 3
    for i, r in enumerate(results):
        assert len(r) > 0, f"Query {i} returned no results"


def test_batch_query_via_core(memory: RTMDKMemory):
    """batch_query via core should work."""
    queries = ["document about topic 0", "document about topic 5"]
    embeddings = [memory.embedder(q) for q in queries]
    results = memory.batch_query(embeddings, top_k=5)
    assert len(results) == 2
    for r in results:
        assert len(r) > 0


def test_batch_pipeline_executor(memory: RTMDKMemory):
    """BatchPipelineExecutor should run batch embed + retrieve."""
    pipeline = memory.build_pipeline()
    executor = BatchPipelineExecutor(pipeline.stages, memory.field)
    queries = ["document about topic 1", "document about topic 3"]
    outputs = executor.run_batch(queries, top_k=5)
    assert len(outputs) == 2
    for out in outputs:
        assert out["total"] > 0
        assert "metrics" in out
