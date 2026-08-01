#!/usr/bin/env python3
"""Stress test RTMDK pipeline with 100K+ synthetic nodes.

Measures:
  - Insert throughput (nodes/sec)
  - Query latency (p50, p95, p99)
  - Memory usage (RSS)
  - Pipeline stage breakdown
  - Cost per query

Usage:
    python scripts/stress_test_pipeline.py --nodes 100000 --queries 1000
    python scripts/stress_test_pipeline.py --nodes 50000 --queries 500 --planner
    python scripts/stress_test_pipeline.py --nodes 100000 --queries 1000 --cost-tracking
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def _make_embedder(dim: int = 384):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


@dataclass
class StressResult:
    nodes: int
    queries: int
    insert_time_sec: float = 0.0
    insert_throughput: float = 0.0
    query_latencies_ms: List[float] = field(default_factory=list)
    memory_mb: float = 0.0
    planner_enabled: bool = False
    cost_tracking_enabled: bool = False
    avg_cost_per_query: float = 0.0
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
        print("STRESS TEST RESULTS")
        print(f"{'='*70}")
        print(f"  Nodes inserted:      {self.nodes:,}")
        print(f"  Insert time:         {self.insert_time_sec:.1f}s")
        print(f"  Insert throughput:   {self.insert_throughput:,.0f} nodes/sec")
        print(f"  Queries run:         {self.queries}")
        print(f"  Latency p50:         {self.latency_p50_ms:.2f}ms")
        print(f"  Latency p95:         {self.latency_p95_ms:.2f}ms")
        print(f"  Latency p99:         {self.latency_p99_ms:.2f}ms")
        print(f"  Memory usage:        {self.memory_mb:.1f}MB")
        print(f"  Planner enabled:     {self.planner_enabled}")
        print(f"  Cost tracking:       {self.cost_tracking_enabled}")
        if self.cost_tracking_enabled:
            print(f"  Avg cost/query:      {self.avg_cost_per_query:.6f}")
        if self.stage_breakdown:
            print(f"  Stage breakdown (ms):")
            for stage, lat in sorted(self.stage_breakdown.items(), key=lambda x: -x[1]):
                print(f"    {stage:12s} {lat:.2f}ms")
        print(f"{'='*70}")

    def to_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "queries": self.queries,
            "insert_time_sec": self.insert_time_sec,
            "insert_throughput": self.insert_throughput,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "memory_mb": self.memory_mb,
            "planner_enabled": self.planner_enabled,
            "cost_tracking_enabled": self.cost_tracking_enabled,
            "avg_cost_per_query": self.avg_cost_per_query,
            "stage_breakdown": self.stage_breakdown,
        }


def get_memory_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def generate_text(dim: int = 384) -> tuple[str, np.ndarray]:
    """Generate random synthetic document + embedding."""
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "memory",
        "resonance",
        "topology",
        "field",
        "vector",
        "search",
        "neural",
        "embedding",
        "semantic",
        "retrieval",
        "knowledge",
        "graph",
    ]
    text = " ".join(random.choices(words, k=random.randint(5, 20)))
    emb = np.random.randn(dim).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return text, emb


def run_stress_test(
    nodes: int,
    queries: int,
    planner: bool = False,
    cost_tracking: bool = False,
    latent_dim: int = 384,
) -> StressResult:
    print(f"Initializing RTMDK with planner={planner}, cost_tracking={cost_tracking}")
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
        learn_projection=False,
        projection_mode="identity",
        pipeline_enabled=True,
        pipeline_planner_enabled=planner,
        pipeline_cost_tracking_enabled=cost_tracking,
    )
    memory = RTMDKMemory(config=cfg, embedder=_make_embedder(latent_dim))

    # Phase 1: Insert nodes
    print(f"Inserting {nodes:,} nodes...")
    gc.collect()
    t0 = time.perf_counter()
    for i in range(nodes):
        text, emb = generate_text(latent_dim)
        memory.add_node(content={"content": text, "query": text}, embedding=emb)
        if i > 0 and i % 10000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i:,} nodes ({i/elapsed:,.0f}/sec)")
    insert_time = time.perf_counter() - t0
    mem_after_insert = get_memory_mb()

    # Phase 2: Query
    print(f"Running {queries} queries...")
    latencies = []
    total_cost = 0.0
    stage_acc: Dict[str, List[float]] = {}

    for i in range(queries):
        query_text, _ = generate_text(latent_dim)
        tq0 = time.perf_counter()
        result = memory.retrieve_nodes_pipeline(query_text, top_k=5)
        latency_ms = (time.perf_counter() - tq0) * 1000
        latencies.append(latency_ms)

        if cost_tracking and result.get("cost"):
            total_cost += result["cost"].get("total_cost", 0.0)

        # Accumulate stage breakdown
        for stage_metric in result.get("metrics", {}).get("stages", []):
            name = stage_metric.get("stage", "unknown")
            stage_acc.setdefault(name, []).append(stage_metric.get("latency_ms", 0.0))

    mem_after_query = get_memory_mb()

    stage_breakdown = {name: float(np.mean(vals)) for name, vals in stage_acc.items() if vals}

    return StressResult(
        nodes=nodes,
        queries=queries,
        insert_time_sec=insert_time,
        insert_throughput=nodes / max(insert_time, 0.001),
        query_latencies_ms=latencies,
        memory_mb=max(mem_after_insert, mem_after_query),
        planner_enabled=planner,
        cost_tracking_enabled=cost_tracking,
        avg_cost_per_query=total_cost / max(queries, 1),
        stage_breakdown=stage_breakdown,
    )


def main():
    parser = argparse.ArgumentParser(description="Stress test RTMDK pipeline")
    parser.add_argument("--nodes", type=int, default=10000, help="Number of nodes to insert")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries to run")
    parser.add_argument("--planner", action="store_true", help="Enable query planner")
    parser.add_argument("--cost-tracking", action="store_true", help="Enable cost tracking")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    result = run_stress_test(
        nodes=args.nodes,
        queries=args.queries,
        planner=args.planner,
        cost_tracking=args.cost_tracking,
        latent_dim=args.dim,
    )
    result.print_summary()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Fail if p99 latency > 100ms per 10K nodes
    expected_p99 = max(1.0, args.nodes / 10000) * 100.0
    if result.latency_p99_ms > expected_p99:
        print(f"\nWARNING: p99 latency {result.latency_p99_ms:.1f}ms exceeds threshold {expected_p99:.1f}ms")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
