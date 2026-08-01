#!/usr/bin/env python3
"""Enterprise stress test: RTMDK with 100K+ nodes and tiered storage v2.

Measures:
  - Insert throughput (nodes/sec)
  - Query latency at scale (p50, p95, p99)
  - Memory usage with tiered storage
  - Tier distribution (hot/warm/cold)
  - Pipeline stage breakdown

Usage:
    python scripts/stress_test_100k.py --nodes 100000 --queries 500
    python scripts/stress_test_100k.py --nodes 50000 --queries 200 --tiered-v2
    python scripts/stress_test_100k.py --nodes 100000 --queries 500 --planner
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def _make_embedder(dim: int = 384):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(dim, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-8 else vec

    return embed


def get_memory_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


@dataclass
class StressResult:
    nodes: int
    queries: int
    tiered_v2: bool
    planner: bool
    insert_time_sec: float = 0.0
    insert_throughput: float = 0.0
    query_latencies_ms: List[float] = field(default_factory=list)
    memory_mb: float = 0.0
    tier_stats: Optional[Dict] = None
    stage_breakdown: Dict[str, float] = field(default_factory=dict)

    @property
    def latency_p50_ms(self) -> float:
        return float(np.percentile(self.query_latencies_ms, 50)) if self.query_latencies_ms else 0.0

    @property
    def latency_p95_ms(self) -> float:
        return float(np.percentile(self.query_latencies_ms, 95)) if self.query_latencies_ms else 0.0

    @property
    def latency_p99_ms(self) -> float:
        return float(np.percentile(self.query_latencies_ms, 99)) if self.query_latencies_ms else 0.0

    def print_summary(self):
        print(f"\n{'='*70}")
        print("ENTERPRISE STRESS TEST RESULTS")
        print(f"{'='*70}")
        print(f"  Nodes inserted:      {self.nodes:,}")
        print(f"  Tiered storage v2:   {self.tiered_v2}")
        print(f"  Query planner:       {self.planner}")
        print(f"  Insert time:         {self.insert_time_sec:.1f}s")
        print(f"  Insert throughput:   {self.insert_throughput:,.0f} nodes/sec")
        print(f"  Queries run:         {self.queries}")
        print(f"  Latency p50:         {self.latency_p50_ms:.2f}ms")
        print(f"  Latency p95:         {self.latency_p95_ms:.2f}ms")
        print(f"  Latency p99:         {self.latency_p99_ms:.2f}ms")
        print(f"  Memory usage:        {self.memory_mb:.1f}MB")
        if self.tier_stats:
            print(f"  Tier distribution:")
            print(f"    Hot:  {self.tier_stats.get('hot_count', 0):,}")
            print(f"    Warm: {self.tier_stats.get('warm_count', 0):,}")
            print(f"    Cold: {self.tier_stats.get('cold_count', 0):,}")
        if self.stage_breakdown:
            print(f"  Stage breakdown (ms):")
            for stage, lat in sorted(self.stage_breakdown.items(), key=lambda x: -x[1]):
                print(f"    {stage:12s} {lat:.2f}ms")
        print(f"{'='*70}")

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "queries": self.queries,
            "tiered_v2": self.tiered_v2,
            "planner": self.planner,
            "insert_time_sec": self.insert_time_sec,
            "insert_throughput": self.insert_throughput,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "memory_mb": self.memory_mb,
            "tier_stats": self.tier_stats,
            "stage_breakdown": self.stage_breakdown,
        }


def run_stress_test(
    nodes: int,
    queries: int,
    tiered_v2: bool = False,
    planner: bool = False,
    latent_dim: int = 384,
) -> StressResult:
    print(f"Initializing RTMDK: nodes={nodes:,}, tiered_v2={tiered_v2}, planner={planner}")
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

    cfg = RTMDKConfig(
        latent_dim=latent_dim,
        embedding_dim=latent_dim,
        max_nodes=nodes + 100,
        top_k=5,
        min_response=0.001,
        bandwidth=1.0,
        phase_coupling=0.0,
        use_hnsw=True,
        hnsw_min_nodes=10,
        bm25_fallback=False,
        learn_projection=False,
        projection_mode="identity",
        pipeline_enabled=True,
        pipeline_planner_enabled=planner,
        pipeline_cost_tracking_enabled=True,
    )

    if tiered_v2:
        cfg.tiered_storage_v2_enabled = True
        cfg.tiered_hot_pct = 0.01  # 1% hot = 1K nodes for 100K
        cfg.tiered_warm_pct = 0.09  # 9% warm = 9K nodes for 100K

    embedder = _make_embedder(latent_dim)
    memory = RTMDKMemory(config=cfg, embedder=embedder)

    # Phase 1: Insert nodes (batched for throughput)
    print(f"Inserting {nodes:,} nodes (batched)...")
    gc.collect()
    t0 = time.perf_counter()
    batch_size = 1000
    batch_embs: List[np.ndarray] = []
    batch_contents: List[Dict] = []
    for i in range(nodes):
        text = f"document about topic {i % 100} aspect {i} keywords {' '.join(random.choices(['neural', 'embedding', 'search', 'vector', 'memory'], k=5))}"
        emb = embedder(text)
        batch_embs.append(emb)
        batch_contents.append({"text": text})
        if len(batch_embs) >= batch_size:
            memory.add_nodes_batch(
                embeddings=np.stack(batch_embs, dtype=np.float32),
                contents=batch_contents,
            )
            batch_embs.clear()
            batch_contents.clear()
            elapsed = time.perf_counter() - t0
            print(f"  {i+1:,} nodes ({(i+1)/elapsed:,.0f}/sec, {get_memory_mb():.0f}MB)")
    if batch_embs:
        memory.add_nodes_batch(
            embeddings=np.stack(batch_embs, dtype=np.float32),
            contents=batch_contents,
        )
        batch_embs.clear()
        batch_contents.clear()
    insert_time = time.perf_counter() - t0
    mem_after_insert = get_memory_mb()

    # Phase 2: Query
    print(f"Running {queries} queries...")
    # Warmup: trigger cache build so first measured query isn't an outlier
    _ = memory.retrieve_nodes_pipeline("warmup query", top_k=5)
    latencies = []
    stage_acc: Dict[str, List[float]] = {}
    query_texts = [f"document about topic {i % 100}" for i in range(queries)]

    for i, q in enumerate(query_texts):
        tq0 = time.perf_counter()
        result = memory.retrieve_nodes_pipeline(q, top_k=5)
        latency_ms = (time.perf_counter() - tq0) * 1000
        latencies.append(latency_ms)

        for stage_metric in result.get("metrics", {}).get("stages", []):
            name = stage_metric.get("stage", "unknown")
            stage_acc.setdefault(name, []).append(stage_metric.get("latency_ms", 0.0))

    mem_after_query = get_memory_mb()
    stage_breakdown = {name: float(np.mean(vals)) for name, vals in stage_acc.items() if vals}

    # Tier stats
    tier_stats = None
    if tiered_v2 and hasattr(memory.field.nodes, "stats"):
        tier_stats = memory.field.nodes.stats()

    return StressResult(
        nodes=nodes,
        queries=queries,
        tiered_v2=tiered_v2,
        planner=planner,
        insert_time_sec=insert_time,
        insert_throughput=nodes / max(insert_time, 0.001),
        query_latencies_ms=latencies,
        memory_mb=max(mem_after_insert, mem_after_query),
        tier_stats=tier_stats,
        stage_breakdown=stage_breakdown,
    )


def main():
    parser = argparse.ArgumentParser(description="Enterprise stress test RTMDK")
    parser.add_argument("--nodes", type=int, default=100_000, help="Number of nodes")
    parser.add_argument("--queries", type=int, default=500, help="Number of queries")
    parser.add_argument("--tiered-v2", action="store_true", help="Enable tiered storage v2")
    parser.add_argument("--planner", action="store_true", help="Enable query planner")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--output", "-o", help="Output JSON")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    result = run_stress_test(
        nodes=args.nodes,
        queries=args.queries,
        tiered_v2=args.tiered_v2,
        planner=args.planner,
        latent_dim=args.dim,
    )
    result.print_summary()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nSaved to {args.output}")

    # Enterprise thresholds
    thresholds = {
        "p99_latency_ms": max(20.0, args.nodes / 10000 * 5.0),
        "memory_mb": max(500, args.nodes * 0.05),
    }
    passed = True
    if result.latency_p99_ms > thresholds["p99_latency_ms"]:
        print(f"\nWARNING: p99 latency {result.latency_p99_ms:.1f}ms exceeds {thresholds['p99_latency_ms']:.1f}ms")
        passed = False
    if result.memory_mb > thresholds["memory_mb"]:
        print(f"\nWARNING: memory {result.memory_mb:.0f}MB exceeds {thresholds['memory_mb']:.0f}MB")
        passed = False
    if passed:
        print("\n[PASS] All enterprise thresholds met")
    else:
        print("\n[FAIL] Some thresholds exceeded")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
