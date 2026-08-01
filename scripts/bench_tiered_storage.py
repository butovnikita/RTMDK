"""Benchmark Tiered Storage v2: RAM, latency, and throughput at scale.

Usage:
    python scripts/bench_tiered_storage.py --nodes 100000 --queries 1000
"""

from __future__ import annotations

import argparse
import gc
import time
import tracemalloc
from pathlib import Path

import numpy as np

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField


def measure_ram() -> int:
    """Return current RSS in MB."""
    gc.collect()
    import os
    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss // (1024 * 1024)


def bench_ingest(field: RTMDKField, n_nodes: int, latent_dim: int) -> dict:
    """Benchmark batch node ingestion."""
    print(f"  Ingesting {n_nodes:,} nodes...")
    start = time.perf_counter()
    for i in range(n_nodes):
        emb = np.random.randn(latent_dim).astype(np.float32)
        field.add_node(emb, {"text": f"document number {i} with some content"})
    elapsed = time.perf_counter() - start
    return {
        "nodes": n_nodes,
        "ingest_time_sec": elapsed,
        "nodes_per_sec": n_nodes / elapsed,
    }


def bench_query(field: RTMDKField, n_queries: int, latent_dim: int) -> dict:
    """Benchmark query latency."""
    print(f"  Running {n_queries:,} queries...")
    latencies = []
    for _ in range(n_queries):
        emb = np.random.randn(latent_dim).astype(np.float32)
        t0 = time.perf_counter()
        field.query(emb, top_k=5)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
    latencies = np.array(latencies)
    return {
        "queries": n_queries,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_mean_ms": float(latencies.mean()),
    }


def bench_promotion(field: RTMDKField, cold_ids: list[str]) -> dict:
    """Benchmark promotion from cold/warm to hot."""
    if not cold_ids:
        return {"promotion_time_ms": 0.0}
    print(f"  Promoting {len(cold_ids)} cold/warm nodes...")
    times = []
    for nid in cold_ids[:100]:  # sample 100
        t0 = time.perf_counter()
        _ = field.nodes[nid]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return {
        "promotion_time_ms": float(np.mean(times)),
        "promotion_sample_size": len(times),
    }


def run_benchmark(n_nodes: int, n_queries: int, latent_dim: int, tmp_dir: Path) -> dict:
    """Run full benchmark for given node count."""
    print(f"\n=== Benchmark: {n_nodes:,} nodes ===")
    cfg = RTMDKConfig(
        latent_dim=latent_dim,
        max_nodes=max(n_nodes + 1000, 10000),
        tiered_storage_v2_enabled=True,
        tiered_storage_path=str(tmp_dir / f"cold_{n_nodes}"),
        tiered_hot_pct=0.01,
        tiered_warm_pct=0.09,
        use_hnsw=False,  # HNSW skews latency; test pure tiered storage
        query_cache_size=0,  # disable cache for fair latency measurement
    )

    field = RTMDKField(config=cfg)

    ram_before = measure_ram()
    ingest_result = bench_ingest(field, n_nodes, latent_dim)
    ram_after = measure_ram()
    ingest_result["ram_mb"] = ram_after
    ingest_result["ram_delta_mb"] = ram_after - ram_before

    stats = field.nodes.stats()
    print(f"  Tier stats: hot={stats['hot_count']}, warm={stats['warm_count']}, cold={stats['cold_count']}")

    query_result = bench_query(field, n_queries, latent_dim)

    # Sample some cold/warm IDs for promotion benchmark
    adapter = field.nodes
    warm_ids = adapter.warm_ids() if hasattr(adapter, "warm_ids") else []
    cold_ids = adapter.cold_ids() if hasattr(adapter, "cold_ids") else []
    promotion_result = bench_promotion(field, cold_ids + warm_ids)

    result = {
        "n_nodes": n_nodes,
        "latent_dim": latent_dim,
        **ingest_result,
        **query_result,
        **promotion_result,
        "tier_stats": stats,
    }

    field.nodes.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark Tiered Storage v2")
    parser.add_argument("--nodes", type=int, default=100_000, help="Total nodes to ingest")
    parser.add_argument("--queries", type=int, default=1_000, help="Number of queries")
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--tmp-dir", type=str, default="./_tiered_bench_tmp", help="Temp directory")
    args = parser.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Run at multiple scales
    scales = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    for scale in scales:
        if scale > args.nodes:
            break
        result = run_benchmark(scale, args.queries, args.latent_dim, tmp_dir)
        results.append(result)
        print(
            f"  RAM: {result['ram_mb']} MB | Ingest: {result['nodes_per_sec']:.0f} nodes/s | Query p50: {result['latency_p50_ms']:.2f} ms"
        )

    # Summary table
    print("\n=== Summary ===")
    print(f"{'Nodes':>12} {'RAM(MB)':>10} {'Ingest/s':>12} {'Query p50':>12} {'Query p99':>12} {'Promote ms':>12}")
    for r in results:
        print(
            f"{r['n_nodes']:>12,} {r['ram_mb']:>10} {r['nodes_per_sec']:>12.0f} "
            f"{r['latency_p50_ms']:>12.2f} {r['latency_p99_ms']:>12.2f} {r.get('promotion_time_ms', 0):>12.2f}"
        )

    # Save JSON
    import json

    out_path = tmp_dir / "tiered_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
