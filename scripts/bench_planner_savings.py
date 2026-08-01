#!/usr/bin/env python3
"""A/B benchmark: Query Planner savings vs baseline pipeline.

Measures real latency and cost savings from query planner stage skipping.

Usage:
    python scripts/bench_planner_savings.py --dataset datasets/comprehensive_500.json --n 200
    python scripts/bench_planner_savings.py --dataset datasets/qa_1000_en.json --n 500 --top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List

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
class PlannerBenchmarkResult:
    queries: int
    baseline_latency_ms: List[float] = field(default_factory=list)
    planned_latency_ms: List[float] = field(default_factory=list)
    baseline_cost: List[float] = field(default_factory=list)
    planned_cost: List[float] = field(default_factory=list)

    @property
    def baseline_p50(self) -> float:
        return float(np.percentile(self.baseline_latency_ms, 50)) if self.baseline_latency_ms else 0.0

    @property
    def planned_p50(self) -> float:
        return float(np.percentile(self.planned_latency_ms, 50)) if self.planned_latency_ms else 0.0

    @property
    def latency_reduction_pct(self) -> float:
        if not self.baseline_p50:
            return 0.0
        return (self.baseline_p50 - self.planned_p50) / self.baseline_p50 * 100

    @property
    def cost_reduction_pct(self) -> float:
        if not self.baseline_cost:
            return 0.0
        base_avg = np.mean(self.baseline_cost)
        plan_avg = np.mean(self.planned_cost)
        return (base_avg - plan_avg) / base_avg * 100

    def print_summary(self):
        print(f"\n{'='*70}")
        print("QUERY PLANNER A/B BENCHMARK")
        print(f"{'='*70}")
        print(f"  Queries:           {self.queries}")
        print(f"  Baseline p50:      {self.baseline_p50:.2f}ms")
        print(f"  Planned p50:       {self.planned_p50:.2f}ms")
        print(f"  Latency reduction: {self.latency_reduction_pct:.1f}%")
        print(f"  Cost reduction:    {self.cost_reduction_pct:.1f}%")
        print(f"{'='*70}")


def run_benchmark(dataset_path: str, n: int, top_k: int, latent_dim: int = 384):
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data)[:n]
    queries = [r["query"] for r in records]

    print(f"Building memory with {len(records)} nodes...")
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
    cfg = RTMDKConfig(
        latent_dim=latent_dim,
        embedding_dim=latent_dim,
        max_nodes=len(records) + 100,
        top_k=top_k,
        min_response=0.001,
        bandwidth=1.0,
        phase_coupling=0.0,
        use_hnsw=True,
        learn_projection=False,
        projection_mode="identity",
        pipeline_enabled=True,
        pipeline_planner_enabled=False,
        pipeline_cost_tracking_enabled=True,
    )
    embedder = _make_embedder(latent_dim)
    memory = RTMDKMemory(config=cfg, embedder=embedder)

    for r in records:
        emb = embedder(r.get("context", r.get("text", "")))
        memory.add_node(content={"text": r.get("context", "")}, embedding=emb)

    result = PlannerBenchmarkResult(queries=len(queries))

    # Baseline: planner disabled
    print("Running baseline (planner disabled)...")
    for q in queries:
        t0 = time.perf_counter()
        res = memory.retrieve_nodes_pipeline(q, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        result.baseline_latency_ms.append(latency_ms)
        result.baseline_cost.append(res.get("cost", {}).get("total_cost", 0.0))

    # Enable planner
    memory.config.pipeline_planner_enabled = True
    print("Running with planner enabled...")
    for q in queries:
        t0 = time.perf_counter()
        res = memory.retrieve_nodes_pipeline(q, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        result.planned_latency_ms.append(latency_ms)
        result.planned_cost.append(res.get("cost", {}).get("total_cost", 0.0))

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark query planner savings")
    parser.add_argument("--dataset", default="datasets/comprehensive_500.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--output", "-o", help="Output JSON file")

    args = parser.parse_args()
    result = run_benchmark(args.dataset, args.n, args.top_k, args.dim)
    result.print_summary()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "queries": result.queries,
                    "baseline_p50_ms": result.baseline_p50,
                    "planned_p50_ms": result.planned_p50,
                    "latency_reduction_pct": result.latency_reduction_pct,
                    "cost_reduction_pct": result.cost_reduction_pct,
                },
                f,
                indent=2,
            )
        print(f"\nSaved to {args.output}")

    # Note: Savings depend on configuration. Max theoretical savings ~40%
    # when rerank + calibrate + explain are enabled and route is "fast".
    # Default config may show smaller savings if expensive stages are disabled.
    sys.exit(0)


if __name__ == "__main__":
    main()
