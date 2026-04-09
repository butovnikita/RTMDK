"""
benchmark_cross_session.py — Cross-Session Recall Benchmark for RTMDK.

Tests whether RTMDK can recall information across different session boundaries.
Also measures session isolation (whether session A data leaks into session B).

Usage:
    python benchmark_cross_session.py [--report cross_session_report.json]
"""

import os
import sys
import json
import argparse
from typing import Dict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


def make_embedder():
    def embed(text: str) -> np.ndarray:
        np.random.seed(42)
        base = np.random.randn(768).astype(np.float32) * 0.01
        tokens = text.lower().split()
        for tok in tokens[:20]:
            np.random.seed(hash(tok + "xsession_seed") % 2**32)
            direction = np.random.randn(768).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            base += direction * 0.5
        return base
    return embed


class CrossSessionBenchmark:
    """Tests cross-session recall and isolation."""

    def __init__(self):
        self.config = RTMDKConfig(
            embedding_dim=768, latent_dim=64, top_k=5,
            min_response=0.01, decay_rate=0.998,
            enable_async=False, causal_topological=False,
            meta_adaptive=False, self_healing=False,
            cross_modal=False, attention_bias=False, use_hnsw=False,
        )
        self.embedder = make_embedder()

    def run(self, n_sessions: int = 5, facts_per_session: int = 20) -> Dict:
        memory = RTMDKMemory(config=self.config, embedder=self.embedder)

        # Store facts in different sessions
        all_facts = {}
        for s in range(n_sessions):
            session_id = f"session_{s}"
            facts = []
            for i in range(facts_per_session):
                fact = f"Session {s} fact {i} about keyword_{s}_{i:03d}"
                query = f"What is session {s} fact {i} about?"
                keyword = f"keyword_{s}_{i:03d}"
                facts.append({"fact": fact, "query": query, "keyword": keyword})
                memory.save_context({"input": fact, "session_id": session_id}, {"output": fact})
                memory.save_context({"input": query, "session_id": session_id}, {"output": fact})
                memory.save_context({"input": keyword, "session_id": session_id}, {"output": fact})
            all_facts[session_id] = facts

        # Step a bit to let memory settle
        for _ in range(20):
            memory.field.step()

        # Test 1: Same-session recall
        print("\n[1] Testing same-session recall...")
        same_session_recalls = {}
        for s in range(n_sessions):
            session_id = f"session_{s}"
            n_correct = 0
            for item in all_facts[session_id][:5]:  # Test first 5
                ctx = memory.load_memory_variables({"input": item["query"], "session_id": session_id})
                if item["keyword"] in ctx.get("rtmdk_context", "").lower():
                    n_correct += 1
            same_session_recalls[session_id] = n_correct / 5
            print(f"  {session_id}: {same_session_recalls[session_id]:.2%}")

        # Test 2: Cross-session recall (query from session A, stored in session B)
        print("\n[2] Testing cross-session recall...")
        cross_session_recalls = {}
        for s in range(n_sessions):
            session_id = f"session_{s}"
            n_correct = 0
            for other_s in range(n_sessions):
                if other_s == s:
                    continue
                other_session = f"session_{other_s}"
                for item in all_facts[other_session][:2]:
                    ctx = memory.load_memory_variables({"input": item["query"], "session_id": session_id})
                    if item["keyword"] in ctx.get("rtmdk_context", "").lower():
                        n_correct += 1
            total_cross = (n_sessions - 1) * 2
            cross_session_recalls[session_id] = n_correct / max(total_cross, 1)
            print(f"  {session_id}: {cross_session_recalls[session_id]:.2%}")

        avg_same = np.mean(list(same_session_recalls.values()))
        avg_cross = np.mean(list(cross_session_recalls.values()))

        return {
            "n_sessions": n_sessions,
            "facts_per_session": facts_per_session,
            "same_session_recall": {k: round(v, 4) for k, v in same_session_recalls.items()},
            "cross_session_recall": {k: round(v, 4) for k, v in cross_session_recalls.items()},
            "avg_same_session_recall": round(float(avg_same), 4),
            "avg_cross_session_recall": round(float(avg_cross), 4),
            "session_isolation_score": round(float(1.0 - avg_cross), 4),
        }


def main():
    parser = argparse.ArgumentParser(description="RTMDK Cross-Session Recall Benchmark")
    parser.add_argument("--n_sessions", type=int, default=5)
    parser.add_argument("--facts_per_session", type=int, default=20)
    parser.add_argument("--report", type=str, default="cross_session_report.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  RTMDK Cross-Session Recall Benchmark")
    print("=" * 60)

    bench = CrossSessionBenchmark()
    report = bench.run(n_sessions=args.n_sessions, facts_per_session=args.facts_per_session)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Avg same-session recall:  {report['avg_same_session_recall']:.2%}")
    print(f"  Avg cross-session recall: {report['avg_cross_session_recall']:.2%}")
    print(f"  Session isolation score:  {report['session_isolation_score']:.2%}")
    print("=" * 60)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {args.report}")


if __name__ == "__main__":
    main()
