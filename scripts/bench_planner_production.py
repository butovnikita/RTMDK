#!/usr/bin/env python3
"""Production A/B benchmark: Query Planner with ALL expensive stages enabled.

This benchmark enables reranker, conformal prediction, and explainability
to measure MAXIMUM theoretical savings from query planner stage skipping.

Usage:
    python scripts/bench_planner_production.py --dataset datasets/comprehensive_500.json --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

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
class ProductionBenchmarkResult:
    queries: int
    baseline_latency_ms: List[float] = field(default_factory=list)
    planned_latency_ms: List[float] = field(default_factory=list)

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

    def print_summary(self):
        print(f"\n{'='*70}")
        print("PRODUCTION QUERY PLANNER A/B BENCHMARK")
        print("(reranker + conformal + explainability ENABLED)")
        print(f"{'='*70}")
        print(f"  Queries:           {self.queries}")
        print(f"  Baseline p50:      {self.baseline_p50:.2f}ms")
        print(f"  Planned p50:       {self.planned_p50:.2f}ms")
        print(f"  Latency reduction: {self.latency_reduction_pct:.1f}%")
        print(f"{'='*70}")
        if self.latency_reduction_pct >= 20:
            print("  [OK] EXCELLENT: Planner saves >20% latency")
        elif self.latency_reduction_pct >= 10:
            print("  [OK] GOOD: Planner saves >10% latency")
        else:
            print("  [WARN] LOW: Planner savings <10% (expected for short queries / standard route)")


def run_benchmark(dataset_path: str, n: int, top_k: int, latent_dim: int = 384):
    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data)[:n]
    queries = [r["query"] for r in records]

    print(f"Building memory with {len(records)} nodes (production config)...")
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
        # ENABLE ALL EXPENSIVE STAGES
        reranker_enabled=True,
        conformal_prediction=True,
        sentence_reranker_enabled=True,
        result_explainability_enabled=True,
    )
    embedder = _make_embedder(latent_dim)
    memory = RTMDKMemory(config=cfg, embedder=embedder)

    for r in records:
        emb = embedder(r.get("context", r.get("text", "")))
        memory.add_node(content={"text": r.get("context", "")}, embedding=emb)

    result = ProductionBenchmarkResult(queries=len(queries))

    # Baseline: planner disabled
    print("Running baseline (planner disabled, all stages active)...")
    for q in queries:
        t0 = time.perf_counter()
        memory.retrieve_nodes_pipeline(q, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        result.baseline_latency_ms.append(latency_ms)

    # Enable planner
    memory.config.pipeline_planner_enabled = True
    print("Running with planner enabled...")
    for q in queries:
        t0 = time.perf_counter()
        memory.retrieve_nodes_pipeline(q, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        result.planned_latency_ms.append(latency_ms)

    return result


def main():
    parser = argparse.ArgumentParser(description="Production planner benchmark")
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
                },
                f,
                indent=2,
            )
        print(f"\nSaved to {args.output}")

    sys.exit(0)


if __name__ == "__main__":
    main()
