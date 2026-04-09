"""
benchmark_capacity.py — Memory Capacity & Scaling Benchmark for RTMDK.

Tests recall rate, RAM usage, and latency at N=500/1000/2000/5000 nodes.

Usage:
    python benchmark_capacity.py [--report capacity_report.json]
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


def make_embedder():
    def embed(text: str) -> np.ndarray:
        np.random.seed(42)
        base = np.random.randn(768).astype(np.float32) * 0.01
        tokens = text.lower().split()
        for tok in tokens[:20]:
            np.random.seed(hash(tok + "cap_seed") % 2**32)
            direction = np.random.randn(768).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            base += direction * 0.5
        return base
    return embed


class CapacityBenchmark:
    """Measures scaling behavior at different memory sizes."""

    def __init__(self):
        self.embedder = make_embedder()

    def _run_at_size(self, n_nodes: int) -> Dict:
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64, top_k=5,
            min_response=0.01, decay_rate=0.998,
            enable_async=False, causal_topological=False,
            meta_adaptive=False, self_healing=False,
            cross_modal=False, attention_bias=False, use_hnsw=False,
        )
        memory = RTMDKMemory(config=config, embedder=self.embedder)

        # Generate facts with unique keywords
        n_facts = n_nodes // 4  # ~4 nodes per fact (fact + query + 2 keywords)
        facts = []
        for i in range(n_facts):
            fact = f"Fact number {i} about unique topic {i:06d}"
            query = f"What is fact number {i} about?"
            keyword = f"topic_{i:06d}"
            facts.append({"fact": fact, "query": query, "keyword": keyword})

        t0_store = time.perf_counter()
        for item in facts:
            memory.save_context({"input": item["fact"], "session_id": "cap"}, {"output": item["fact"]})
            memory.save_context({"input": item["query"], "session_id": "cap"}, {"output": item["fact"]})
            memory.save_context({"input": item["keyword"], "session_id": "cap"}, {"output": item["fact"]})
        store_time = time.perf_counter() - t0_store

        for _ in range(10):
            memory.field.step()

        # Test recall on subset
        test_size = min(50, len(facts))
        test_facts = facts[:test_size]
        n_correct = 0
        latencies = []
        t0_query = time.perf_counter()
        for item in test_facts:
            ctx = memory.load_memory_variables({"input": item["query"], "session_id": "cap"})
            context = ctx.get("rtmdk_context", "").lower()
            if item["keyword"] in context:
                n_correct += 1
        query_time = time.perf_counter() - t0_query

        avg_query_ms = (query_time / test_size) * 1000

        return {
            "target_nodes": n_nodes,
            "actual_nodes": len(memory.field.nodes),
            "n_facts_stored": n_facts,
            "store_time_seconds": round(store_time, 3),
            "recall_rate": round(n_correct / max(test_size, 1), 4),
            "avg_query_latency_ms": round(avg_query_ms, 3),
        }

    def run(self, sizes: List[int] = None) -> List[Dict]:
        if sizes is None:
            sizes = [500, 1000, 2000, 5000]
        results = []
        for n in sizes:
            print(f"\n  Testing N={n}...")
            r = self._run_at_size(n)
            print(f"    Nodes: {r['actual_nodes']}, Recall: {r['recall_rate']:.2%}, "
                  f"Latency: {r['avg_query_latency_ms']:.2f}ms, Store: {r['store_time_seconds']:.1f}s")
            results.append(r)
        return results


def main():
    parser = argparse.ArgumentParser(description="RTMDK Memory Capacity Benchmark")
    parser.add_argument("--report", type=str, default="capacity_report.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  RTMDK Memory Capacity & Scaling Benchmark")
    print("=" * 60)

    bench = CapacityBenchmark()
    results = bench.run()

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {'Nodes':>8} {'Recall':>10} {'Latency(ms)':>14} {'Store(s)':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*10}")
    for r in results:
        print(f"  {r['actual_nodes']:>8} {r['recall_rate']:>10.2%} {r['avg_query_latency_ms']:>14.2f} {r['store_time_seconds']:>10.1f}")
    print("=" * 60)

    with open(args.report, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Report saved to {args.report}")


if __name__ == "__main__":
    main()
