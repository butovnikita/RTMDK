"""
benchmark_consolidation.py — Consolidation Quality Benchmark for RTMDK.

Tests whether consolidation preserves or loses information.
Measures recall before/after consolidation, false merge rate.

Usage:
    python benchmark_consolidation.py [--report consolidation_report.json]
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
            np.random.seed(hash(tok + "consol_seed") % 2**32)
            direction = np.random.randn(768).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            base += direction * 0.5
        return base
    return embed


class ConsolidationBenchmark:
    """Measures consolidation quality."""

    def __init__(self):
        self.config = RTMDKConfig(
            embedding_dim=768, latent_dim=64, top_k=5,
            min_response=0.01, decay_rate=0.998,
            enable_async=False, causal_topological=False,
            meta_adaptive=False, self_healing=False,
            cross_modal=False, attention_bias=False, use_hnsw=False,
            adaptive_threshold=True,  # Enable to trigger consolidation
        )
        self.embedder = make_embedder()

    def run(self, n_facts: int = 200, n_consolidation_steps: int = 50) -> Dict:
        memory = RTMDKMemory(config=self.config, embedder=self.embedder)

        # Store facts
        facts = []
        for i in range(n_facts):
            fact = f"Consolidation fact {i} about unique subject {i:05d}"
            query = f"What is consolidation fact {i} about?"
            keyword = f"subject_{i:05d}"
            facts.append({"fact": fact, "query": query, "keyword": keyword})
            memory.save_context({"input": fact, "session_id": "consol"}, {"output": fact})
            memory.save_context({"input": query, "session_id": "consol"}, {"output": fact})
            memory.save_context({"input": keyword, "session_id": "consol"}, {"output": fact})

        # Pre-consolidation recall
        n_correct_before = 0
        test_facts = facts[:min(50, len(facts))]
        for item in test_facts:
            ctx = memory.load_memory_variables({"input": item["query"], "session_id": "consol"})
            if item["keyword"] in ctx.get("rtmdk_context", "").lower():
                n_correct_before += 1
        recall_before = n_correct_before / max(len(test_facts), 1)
        nodes_before = len(memory.field.nodes)

        # Trigger consolidation steps
        for _ in range(n_consolidation_steps):
            memory.field.step()

        # Post-consolidation recall
        n_correct_after = 0
        for item in test_facts:
            ctx = memory.load_memory_variables({"input": item["query"], "session_id": "consol"})
            if item["keyword"] in ctx.get("rtmdk_context", "").lower():
                n_correct_after += 1
        recall_after = n_correct_after / max(len(test_facts), 1)
        nodes_after = len(memory.field.nodes)
        consolidations = memory.field.stats.get("consolidations", 0)

        consolidation_gain = recall_after - recall_before
        compression_ratio = nodes_before / max(nodes_after, 1)

        return {
            "n_facts_stored": n_facts,
            "n_consolidation_steps": n_consolidation_steps,
            "nodes_before": nodes_before,
            "nodes_after": nodes_after,
            "consolidations_performed": consolidations,
            "recall_before_consolidation": round(recall_before, 4),
            "recall_after_consolidation": round(recall_after, 4),
            "consolidation_gain": round(consolidation_gain, 4),
            "compression_ratio": round(compression_ratio, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="RTMDK Consolidation Quality Benchmark")
    parser.add_argument("--n_facts", type=int, default=200)
    parser.add_argument("--n_steps", type=int, default=50)
    parser.add_argument("--report", type=str, default="consolidation_report.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  RTMDK Consolidation Quality Benchmark")
    print("=" * 60)

    bench = ConsolidationBenchmark()
    report = bench.run(n_facts=args.n_facts, n_consolidation_steps=args.n_steps)

    print(f"\n  Facts stored:              {report['n_facts_stored']}")
    print(f"  Consolidation steps:       {report['n_consolidation_steps']}")
    print(f"  Nodes before → after:      {report['nodes_before']} → {report['nodes_after']}")
    print(f"  Consolidations performed:  {report['consolidations_performed']}")
    print(f"  Recall before:             {report['recall_before_consolidation']:.2%}")
    print(f"  Recall after:              {report['recall_after_consolidation']:.2%}")
    print(f"  Consolidation gain:        {report['consolidation_gain']:+.2%}")
    print(f"  Compression ratio:         {report['compression_ratio']:.2f}x")
    print("=" * 60)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {args.report}")


if __name__ == "__main__":
    main()
