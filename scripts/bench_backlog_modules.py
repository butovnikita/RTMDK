"""Smoke benchmark for backlog modules: measure latency impact.

Compares baseline retrieval vs retrieval with all new backlog modules enabled.
"""

import time
import numpy as np
from rtmdk.memory.core import RTMDKMemory, RTMDKConfig


def embed(text):
    return np.random.randn(384).astype(np.float32)


def _make_mem(config):
    mem = RTMDKMemory(config=config, embedder=embed)
    for i in range(100):
        mem.add_node(
            embed(f"doc{i}"),
            {"text": f"Document number {i} about various topics and subjects"},
        )
    return mem


def benchmark(name, mem, n_queries=50):
    queries = [f"query about topic {i}" for i in range(n_queries)]
    t0 = time.perf_counter()
    for q in queries:
        mem.load_memory_variables({"input": q, "session_id": "bench"})
    latency_ms = (time.perf_counter() - t0) * 1000 / n_queries
    print(f"  {name}: {latency_ms:.3f} ms/query")
    return latency_ms


def main():
    print("Backlog modules smoke benchmark")
    print("-" * 40)

    # Baseline
    cfg_base = RTMDKConfig(embedding_dim=384, latent_dim=64)
    mem_base = _make_mem(cfg_base)
    base_lat = benchmark("Baseline", mem_base)

    # With observability + cache
    cfg_obs = RTMDKConfig(embedding_dim=384, latent_dim=64)
    cfg_obs.sot.observability_enabled = True
    cfg_obs.sot.engram_cache_enabled = True
    mem_obs = _make_mem(cfg_obs)
    obs_lat = benchmark("+Observability+Cache", mem_obs)

    # With RAG quality modules
    cfg_rag = RTMDKConfig(embedding_dim=384, latent_dim=64)
    cfg_rag.sot.sentence_reranker_enabled = True
    cfg_rag.sot.query_decomposition_enabled = True
    cfg_rag.sot.feedback_loop_enabled = True
    mem_rag = _make_mem(cfg_rag)
    rag_lat = benchmark("+RAG Quality", mem_rag)

    # All modules
    cfg_all = RTMDKConfig(embedding_dim=384, latent_dim=64)
    cfg_all.sot.observability_enabled = True
    cfg_all.sot.engram_cache_enabled = True
    cfg_all.sot.sentence_reranker_enabled = True
    cfg_all.sot.query_decomposition_enabled = True
    cfg_all.sot.feedback_loop_enabled = True
    mem_all = _make_mem(cfg_all)
    all_lat = benchmark("All modules", mem_all)

    print("-" * 40)
    print(f"Overhead vs baseline:")
    print(f"  Observability+Cache: +{((obs_lat / base_lat - 1) * 100):.1f}%")
    print(f"  RAG Quality:         +{((rag_lat / base_lat - 1) * 100):.1f}%")
    print(f"  All modules:         +{((all_lat / base_lat - 1) * 100):.1f}%")

    # Telemetry stats check
    if mem_obs.metrics is not None:
        stats = mem_obs.metrics.query_latency.percentiles()
        print(f"\nTelemetry p50: {stats['p50']:.3f}ms")

    # Cache check
    if mem_obs.engram_cache is not None:
        print(f"Engram cache size: {len(mem_obs.engram_cache)} entries")


if __name__ == "__main__":
    main()
