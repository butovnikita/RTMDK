#!/usr/bin/env python3
"""Production benchmark: pipeline vs legacy retrieval.

Runs systematic benchmarks across datasets and configurations,
saving results for regression tracking.

Usage:
    python scripts/bench_pipeline_production.py --dataset datasets/qa_1000_en.json
    python scripts/bench_pipeline_production.py --all-datasets --output benchmarks/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def _hash_embed(text: str, dim: int = 64) -> np.ndarray:
    h = hash(text) % (2 ** 32)
    rng = np.random.default_rng(h)
    return rng.standard_normal(dim, dtype=np.float32)


@dataclass
class BenchmarkResult:
    dataset: str
    method: str  # "legacy" or "pipeline"
    nodes: int
    queries: int
    top_k: int
    total_time_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    errors: int = 0
    recall_at_k: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def compute_stats(self):
        if self.latencies:
            self.mean_latency_ms = statistics.mean(self.latencies)
            self.p95_latency_ms = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if len(self.latencies) > 1 else self.latencies[0]


def _load_dataset(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def _build_memory(nodes_data: List[Dict], dim: int = 64) -> RTMDKMemory:
    cfg = RTMDKConfig(latent_dim=dim, embedding_dim=dim, use_hnsw=False)
    mem = RTMDKMemory(config=cfg, embedder=lambda x: _hash_embed(x, dim))
    for item in nodes_data:
        text = item.get("text", item.get("content", item.get("query", "")))
        emb = _hash_embed(text, dim)
        mem.add_node(embedding=emb, content={"text": text})
    return mem


def _run_benchmark(
    mem: RTMDKMemory,
    queries: List[str],
    ground_truth: List[List[str]],
    top_k: int,
    use_pipeline: bool,
) -> BenchmarkResult:
    result = BenchmarkResult(
        dataset="unknown",
        method="pipeline" if use_pipeline else "legacy",
        nodes=len(mem.field.nodes),
        queries=len(queries),
        top_k=top_k,
    )

    t0_total = time.perf_counter()
    for i, query_text in enumerate(queries):
        emb = _hash_embed(query_text, 64)
        t0 = time.perf_counter()
        try:
            if use_pipeline:
                out = mem.retrieve_nodes_pipeline(query_text, embedding=emb, top_k=top_k)
                results = [nid for nid, _, _ in out["results"]]
            else:
                raw = mem.retrieve_nodes(query_text, embedding=emb, top_k=top_k)
                results = [nid for nid, _, _ in raw]

            latency = (time.perf_counter() - t0) * 1000
            result.latencies.append(latency)

            # Compute recall@k if ground truth available
            if ground_truth and i < len(ground_truth):
                gt_set = set(ground_truth[i][:top_k])
                if gt_set:
                    hits = len(gt_set & set(results[:top_k]))
                    result.recall_at_k += hits / len(gt_set)
        except Exception:
            result.errors += 1

    result.total_time_ms = (time.perf_counter() - t0_total) * 1000
    result.compute_stats()
    if result.queries > 0:
        result.recall_at_k /= result.queries

    return result


def main():
    parser = argparse.ArgumentParser(description="Production benchmark: pipeline vs legacy")
    parser.add_argument("--dataset", "-d", type=str, default="",
                        help="Path to dataset JSON")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run on all datasets in datasets/")
    parser.add_argument("--output", "-o", type=str, default="benchmark_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="top_k for retrieval")
    parser.add_argument("--queries", "-q", type=int, default=100,
                        help="Number of queries to run")

    args = parser.parse_args()

    datasets = []
    if args.all_datasets:
        ds_dir = Path("datasets")
        if ds_dir.exists():
            datasets = [str(p) for p in ds_dir.glob("*.json")]
    elif args.dataset:
        datasets = [args.dataset]

    if not datasets:
        print("No datasets found. Use --dataset or --all-datasets")
        sys.exit(1)

    all_results: List[Dict] = []

    for ds_path in datasets:
        print(f"\nBenchmarking: {ds_path}")
        try:
            data = _load_dataset(ds_path)
            # Use first N items as nodes, next M as queries
            nodes_data = data[:min(len(data), 500)]
            queries_data = data[500:500 + args.queries] if len(data) > 500 else data[:args.queries]
            queries = [q.get("text", q.get("query", "")) for q in queries_data]

            mem = _build_memory(nodes_data)

            # Ground truth: for simplicity, use legacy as baseline
            print("  Building ground truth with legacy retrieval...")
            ground_truth = []
            for q in queries:
                emb = _hash_embed(q, 64)
                raw = mem.retrieve_nodes(q, embedding=emb, top_k=args.top_k)
                ground_truth.append([nid for nid, _, _ in raw])

            print(f"  Running pipeline benchmark ({args.top_k})...")
            pipe_result = _run_benchmark(mem, queries, ground_truth, args.top_k, use_pipeline=True)
            pipe_result.dataset = ds_path

            print(f"  Running legacy benchmark ({args.top_k})...")
            legacy_result = _run_benchmark(mem, queries, ground_truth, args.top_k, use_pipeline=False)
            legacy_result.dataset = ds_path

            print(f"  Pipeline:  recall@{args.top_k}={pipe_result.recall_at_k:.3f} "
                  f"p95={pipe_result.p95_latency_ms:.2f}ms")
            print(f"  Legacy:    recall@{args.top_k}={legacy_result.recall_at_k:.3f} "
                  f"p95={legacy_result.p95_latency_ms:.2f}ms")

            all_results.append(asdict(pipe_result))
            all_results.append(asdict(legacy_result))

        except Exception as exc:
            print(f"  ERROR: {exc}")
            import traceback
            traceback.print_exc()

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
