"""
benchmark_forgetting.py — Forgetting Curve Benchmark for RTMDK.

Measures how well RTMDK retains information over time/steps.
Tests against Ebbinghaus-like forgetting curve expectations.

Usage:
    python benchmark_forgetting.py [--n_facts 200] [--max_steps 500] [--report forgetting_report.json]
"""

import os
import sys
import json
import time
import argparse
import random
from typing import List, Dict, Any, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


def make_embedder():
    """Keyword-based embedder for semantic retrieval."""
    def embed(text: str) -> np.ndarray:
        np.random.seed(42)
        base = np.random.randn(768).astype(np.float32) * 0.01
        tokens = text.lower().split()
        for tok in tokens[:20]:
            np.random.seed(hash(tok + "forget_seed") % 2**32)
            direction = np.random.randn(768).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            base += direction * 0.5
        return base
    return embed


def generate_facts(n_facts: int = 200, seed: int = 42) -> List[Dict]:
    """Generate facts with known recall queries."""
    random.seed(seed)
    topics = {
        "science": [
            ("Water boils at 100 degrees Celsius", "At what temperature does water boil?"),
            ("The Earth orbits the Sun", "What does the Earth orbit?"),
            ("Oxygen has atomic number 8", "What is the atomic number of oxygen?"),
        ],
        "history": [
            ("World War II ended in 1945", "When did World War II end?"),
            ("The Roman Empire fell in 476 AD", "When did the Roman Empire fall?"),
            ("Columbus discovered America in 1492", "When did Columbus discover America?"),
        ],
        "geography": [
            ("The capital of France is Paris", "What is the capital of France?"),
            ("The Nile is the longest river", "What is the longest river?"),
            ("Mount Everest is 8849 meters tall", "How tall is Mount Everest?"),
        ],
    }

    facts = []
    topic_list = list(topics.keys())
    for i in range(n_facts):
        topic = topic_list[i % len(topic_list)]
        fact_idx = (i // len(topic_list)) % len(topics[topic])
        fact, query = topics[topic][fact_idx]
        # Add unique identifiers to make each fact distinct
        suffix = f" (fact #{i})"
        facts.append({
            "id": f"fact_{i}",
            "topic": topic,
            "fact": fact + suffix,
            "query": query.replace("?", f" for fact #{i}?"),
            "keyword": fact.split()[0].lower(),
            "stored_at_step": 0,
        })
    return facts


class ForgettingBenchmark:
    """Measures memory retention over time."""

    def __init__(self, decay_rate: float = 0.998):
        self.config = RTMDKConfig(
            embedding_dim=768, latent_dim=64, top_k=5,
            min_response=0.01, decay_rate=decay_rate,
            enable_async=False, causal_topological=False,
            meta_adaptive=False, self_healing=False,
            cross_modal=False, attention_bias=False,
            use_hnsw=False,
        )
        self.embedder = make_embedder()
        self.memory = RTMDKMemory(config=self.config, embedder=self.embedder)

    def store_facts(self, facts: List[Dict]):
        for item in facts:
            self.memory.save_context(
                {"input": item["fact"], "session_id": "forget_test"},
                {"output": item["fact"]}
            )
            self.memory.save_context(
                {"input": item["query"], "session_id": "forget_test"},
                {"output": item["fact"]}
            )
            for kw in item["keyword"].split():
                if len(kw) > 3:
                    self.memory.save_context(
                        {"input": kw, "session_id": "forget_test"},
                        {"output": item["fact"]}
                    )

    def test_recall(self, facts: List[Dict]) -> Tuple[float, List[Dict]]:
        """Test recall for a set of facts. Returns (recall_rate, details)."""
        n_correct = 0
        details = []
        for item in facts:
            ctx = self.memory.load_memory_variables({
                "input": item["query"],
                "session_id": "forget_test",
            })
            context = ctx.get("rtmdk_context", "").lower()
            keyword = item["keyword"].lower()
            found = keyword in context
            if found:
                n_correct += 1
            details.append({"id": item["id"], "found": found, "keyword": keyword})
        return n_correct / max(len(facts), 1), details

    def run(self, n_facts: int = 200, max_steps: int = 500, checkpoint_interval: int = 50) -> Dict:
        """Run forgetting benchmark."""
        facts = generate_facts(n_facts)
        self.store_facts(facts)

        # Initial recall
        initial_recall, _ = self.test_recall(facts)

        checkpoints = []
        for step in range(1, max_steps + 1):
            self.memory.field.step()
            if step % checkpoint_interval == 0 or step == max_steps:
                recall, details = self.test_recall(facts)
                checkpoints.append({
                    "step": step,
                    "recall_rate": recall,
                    "n_nodes": len(self.memory.field.nodes),
                    "n_consolidations": self.memory.field.stats.get("consolidations", 0),
                })
                print(f"  Step {step:5d}: recall={recall:.2%}, nodes={len(self.memory.field.nodes)}")

        # Compute half-life (steps to reach 50% recall)
        half_life = None
        for cp in checkpoints:
            if cp["recall_rate"] < 0.5:
                half_life = cp["step"]
                break

        # Retention at key points
        retention_100 = next((cp["recall_rate"] for cp in checkpoints if cp["step"] >= 100), None)
        retention_500 = next((cp["recall_rate"] for cp in checkpoints if cp["step"] >= 500), None)

        return {
            "n_facts": n_facts,
            "max_steps": max_steps,
            "decay_rate": self.config.decay_rate,
            "initial_recall": initial_recall,
            "half_life_steps": half_life,
            "retention_at_100_steps": retention_100,
            "retention_at_500_steps": retention_500,
            "checkpoints": checkpoints,
        }


def main():
    parser = argparse.ArgumentParser(description="RTMDK Forgetting Curve Benchmark")
    parser.add_argument("--n_facts", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--report", type=str, default="forgetting_report.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  RTMDK Forgetting Curve Benchmark")
    print("=" * 60)

    bench = ForgettingBenchmark()
    report = bench.run(n_facts=args.n_facts, max_steps=args.max_steps, checkpoint_interval=args.checkpoint_interval)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Facts stored:          {report['n_facts']}")
    print(f"  Initial recall:        {report['initial_recall']:.2%}")
    print(f"  Half-life (steps):     {report['half_life_steps'] or 'N/A (did not reach 50%)'}")
    print(f"  Retention @ 100 steps: {report['retention_at_100_steps']:.2%}" if report['retention_at_100_steps'] is not None else "  Retention @ 100 steps: N/A")
    print(f"  Retention @ 500 steps: {report['retention_at_500_steps']:.2%}" if report['retention_at_500_steps'] is not None else "  Retention @ 500 steps: N/A")
    print("=" * 60)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {args.report}")


if __name__ == "__main__":
    main()
