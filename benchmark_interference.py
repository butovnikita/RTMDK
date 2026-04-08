"""
benchmark_interference.py — Memory Interference Benchmark for RTMDK.

Tests whether adding new information corrupts old memories.
Measures catastrophic forgetting and topic bleed.

Usage:
    python benchmark_interference.py [--n_facts_per_topic 50] [--report interference_report.json]
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


def make_embedder():
    def embed(text: str) -> np.ndarray:
        np.random.seed(42)
        base = np.random.randn(768).astype(np.float32) * 0.01
        tokens = text.lower().split()
        for tok in tokens[:20]:
            np.random.seed(hash(tok + "interfere_seed") % 2**32)
            direction = np.random.randn(768).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            base += direction * 0.5
        return base
    return embed


TOPICS = {
    "cooking": {
        "facts": [
            "Salt enhances flavor in soup dishes",
            "Baking soda makes cookies rise",
            "Olive oil is good for salad dressing",
            "Garlic should be minced before cooking",
            "Butter melts at 32 degrees Celsius",
        ],
        "queries": [
            "What enhances flavor in soup?",
            "What makes cookies rise?",
            "What is good for salad dressing?",
            "How should garlic be prepared?",
            "At what temperature does butter melt?",
        ],
        "keywords": ["salt", "baking", "olive", "garlic", "butter"],
    },
    "sports": {
        "facts": [
            "Basketball teams have 5 players on court",
            "A marathon is 42.195 kilometers long",
            "Tennis scoring uses 15-30-40-game system",
            "Soccer matches last 90 minutes",
            "Swimming pools are 50 meters long for Olympics",
        ],
        "queries": [
            "How many basketball players on court?",
            "How long is a marathon?",
            "How does tennis scoring work?",
            "How long are soccer matches?",
            "How long are Olympic swimming pools?",
        ],
        "keywords": ["basketball", "marathon", "tennis", "soccer", "swimming"],
    },
    "technology": {
        "facts": [
            "Python was created by Guido van Rossum",
            "TCP/IP has 4 layers in the model",
            "RAM stands for Random Access Memory",
            "HTML is used for web page structure",
            "SQL is used for database queries",
        ],
        "queries": [
            "Who created Python?",
            "How many TCP/IP layers?",
            "What does RAM stand for?",
            "What is HTML used for?",
            "What is SQL used for?",
        ],
        "keywords": ["python", "tcp", "ram", "html", "sql"],
    },
}


class InterferenceBenchmark:
    """Measures memory interference when adding new topics."""

    def __init__(self):
        self.config = RTMDKConfig(
            embedding_dim=768, latent_dim=64, top_k=5,
            min_response=0.01, decay_rate=0.998,
            enable_async=False, causal_topological=False,
            meta_adaptive=False, self_healing=False,
            cross_modal=False, attention_bias=False, use_hnsw=False,
        )
        self.embedder = make_embedder()
        self.memory = RTMDKMemory(config=self.config, embedder=self.embedder)

    def _store_topic(self, topic_name: str, facts: List[str], queries: List[str], keywords: List[str]):
        for i, (fact, query) in enumerate(zip(facts, queries)):
            self.memory.save_context(
                {"input": fact, "session_id": topic_name},
                {"output": fact}
            )
            self.memory.save_context(
                {"input": query, "session_id": topic_name},
                {"output": fact}
            )
            for kw in keywords:
                if len(kw) > 2:
                    self.memory.save_context(
                        {"input": kw, "session_id": topic_name},
                        {"output": fact}
                    )

    def _test_topic(self, topic_name: str, queries: List[str], facts: List[str], keywords: List[str]) -> float:
        n_correct = 0
        for i, query in enumerate(queries):
            ctx = self.memory.load_memory_variables({
                "input": query,
                "session_id": topic_name,
            })
            context = ctx.get("rtmdk_context", "").lower()
            keyword = keywords[i].lower()
            if keyword in context:
                n_correct += 1
        return n_correct / max(len(queries), 1)

    def run(self, n_per_topic: int = 50) -> Dict:
        """Run interference benchmark."""
        topic_names = list(TOPICS.keys())

        # Phase 1: Store all topics
        print("\n[1] Storing all topics...")
        for t_name in topic_names:
            t = TOPICS[t_name]
            # Repeat facts n_per_topic times with variations
            for repeat in range(n_per_topic):
                facts = [f"{f} (variant {repeat})" for f in t["facts"]]
                self._store_topic(t_name, facts, t["queries"], t["keywords"])
            self.memory.field.step()

        # Phase 2: Measure initial recall per topic
        print("\n[2] Initial recall per topic...")
        initial_recalls = {}
        for t_name in topic_names:
            t = TOPICS[t_name]
            recall = self._test_topic(t_name, t["queries"], t["facts"], t["keywords"])
            initial_recalls[t_name] = recall
            print(f"  {t_name}: {recall:.2%}")

        # Phase 3: Add interfering topic (noise)
        print("\n[3] Adding interfering noise (500 random facts)...")
        for i in range(500):
            self.memory.save_context(
                {"input": f"Random noise fact number {i} about unrelated topic {i % 20}", "session_id": "noise"},
                {"output": f"Random noise fact {i}"}
            )
            if i % 100 == 0:
                self.memory.field.step()

        # Phase 4: Measure post-interference recall
        print("\n[4] Post-interference recall...")
        post_recalls = {}
        for t_name in topic_names:
            t = TOPICS[t_name]
            recall = self._test_topic(t_name, t["queries"], t["facts"], t["keywords"])
            post_recalls[t_name] = recall
            print(f"  {t_name}: {recall:.2%} (was {initial_recalls[t_name]:.2%})")

        # Compute metrics
        interference_ratios = []
        for t_name in topic_names:
            initial = initial_recalls[t_name]
            post = post_recalls[t_name]
            if initial > 0:
                interference_ratios.append((initial - post) / initial)
            else:
                interference_ratios.append(0.0)

        avg_interference = np.mean(interference_ratios)
        topic_bleed = {t: post_recalls[t] - initial_recalls[t] for t in topic_names}

        return {
            "n_facts_per_topic": n_per_topic,
            "n_total_stored": n_per_topic * len(topic_names) * 7,  # facts + queries + keywords
            "n_noise_facts": 500,
            "initial_recall": initial_recalls,
            "post_interference_recall": post_recalls,
            "avg_interference_ratio": float(avg_interference),
            "topic_bleed": {k: float(v) for k, v in topic_bleed.items()},
            "catastrophic_forgetting": any(post < 0.2 for post in post_recalls.values()),
        }


def main():
    parser = argparse.ArgumentParser(description="RTMDK Memory Interference Benchmark")
    parser.add_argument("--n_facts_per_topic", type=int, default=50)
    parser.add_argument("--report", type=str, default="interference_report.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  RTMDK Memory Interference Benchmark")
    print("=" * 60)

    bench = InterferenceBenchmark()
    report = bench.run(n_per_topic=args.n_facts_per_topic)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Facts per topic:       {report['n_facts_per_topic']}")
    print(f"  Total facts stored:    {report['n_total_stored']}")
    print(f"  Noise facts added:     {report['n_noise_facts']}")
    print(f"  Avg interference ratio:{report['avg_interference_ratio']:.2%}")
    print(f"  Catastrophic forgetting: {'YES' if report['catastrophic_forgetting'] else 'No'}")
    for t in TOPICS:
        print(f"  {t}: {report['initial_recall'][t]:.2%} → {report['post_interference_recall'][t]:.2%}")
    print("=" * 60)

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {args.report}")


if __name__ == "__main__":
    main()
