"""Performance regression benchmarks for RTMDK.

Run with: python -m pytest tests/bench_latency.py -v --tb=short
These tests FAIL if p99 latency exceeds baseline thresholds.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField


def _build_field(n_nodes: int, latent_dim: int = 32):
    import os
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
    cfg = RTMDKConfig(latent_dim=latent_dim, max_nodes=n_nodes + 10, top_k=10)
    field = RTMDKField(cfg)
    rng = np.random.default_rng(42)
    for i in range(n_nodes):
        emb = rng.standard_normal(latent_dim).astype(np.float32)
        field.add_node(embedding=emb, content={"text": f"node_{i}"})
    return field


class TestQueryLatency:
    """Benchmark query latency at different node counts."""

    BASELINE_P99_MS = {
        1000: 100.0,
        10000: 600.0,
    }

    @pytest.mark.parametrize("n_nodes", [1000, 10000])
    def test_query_latency_p99(self, n_nodes):
        field = _build_field(n_nodes)
        rng = np.random.default_rng(123)
        query_emb = rng.standard_normal(field.cfg.latent_dim).astype(np.float32)

        # Warmup
        for _ in range(10):
            field.query(query_emb, top_k=10)

        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            field.query(query_emb, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99) - 1]
        threshold = self.BASELINE_P99_MS.get(n_nodes, 100.0)

        assert p99 < threshold, f"p99 query latency {p99:.2f}ms exceeds threshold {threshold}ms for {n_nodes} nodes"


class TestBatchIngestLatency:
    """Benchmark batch ingestion throughput."""

    BASELINE_MS_PER_NODE = 2.0  # max 2ms per node for batch ingest

    def test_batch_ingest_100_nodes(self):
        cfg = RTMDKConfig(latent_dim=32, max_nodes=200, top_k=5)
        field = RTMDKField(cfg)
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((100, 32)).astype(np.float32)
        contents = [{"text": f"doc_{i}"} for i in range(100)]

        t0 = time.perf_counter()
        field.add_nodes_batch(embs, contents)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ms_per_node = elapsed_ms / 100
        assert ms_per_node < self.BASELINE_MS_PER_NODE, (
            f"Batch ingest {ms_per_node:.2f}ms/node exceeds threshold {self.BASELINE_MS_PER_NODE}ms/node"
        )
