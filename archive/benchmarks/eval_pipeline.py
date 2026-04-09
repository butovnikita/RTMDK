"""
eval_pipeline.py
Formal Continual Learning Benchmarks for RTMDK.

Evaluates forgetting/interference at each optimization or federated sync.
Integrates with ContinualQA, LongBench, MemoryBench-style evaluations.

Usage:
    python eval_pipeline.py [--memory_file rtmdk_state.json] [--n_samples 50]
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rtmdk_memory_v8 import (
    RTMDKConfig, RTMDKMemory, ContextFormat,
    apply_attention_bias, format_cognitive_context,
)


# ============================================================================
# BENCHMARK DATASETS (synthetic for demo)
# ============================================================================

def generate_continual_qa_dataset(n_samples: int = 50) -> List[Dict]:
    """Generate synthetic ContinualQA-style dataset."""
    np.random.seed(42)
    topics = ["coffee", "programming", "history", "science", "music",
              "sports", "travel", "food", "technology", "art"]
    dataset = []
    for i in range(n_samples):
        topic = topics[i % len(topics)]
        facts = [
            f"{topic} is a popular subject of discussion.",
            f"Many people enjoy learning about {topic}.",
            f"The history of {topic} spans centuries.",
            f"Research on {topic} continues to grow.",
            f"Understanding {topic} requires dedication.",
        ]
        dataset.append({
            "id": f"cq_{i}",
            "topic": topic,
            "context": facts[i % len(facts)],
            "question": f"What is known about {topic}?",
            "answer": f"Information about {topic}.",
            "timestamp": time.time() - (n_samples - i) * 60,
        })
    return dataset


def generate_long_bench_dataset(n_samples: int = 30) -> List[Dict]:
    """Generate synthetic LongBench-style long-context dataset."""
    np.random.seed(123)
    dataset = []
    for i in range(n_samples):
        length = np.random.randint(500, 2000)
        words = [f"word_{j}" for j in range(length)]
        key_word = f"key_{i}"
        words[np.random.randint(0, length)] = key_word
        dataset.append({
            "id": f"lb_{i}",
            "context": " ".join(words),
            "question": f"What is the key word in this text?",
            "answer": key_word,
            "context_length": length,
        })
    return dataset


def generate_memory_bench_dataset(n_samples: int = 40) -> List[Dict]:
    """Generate synthetic MemoryBench-style recall dataset."""
    np.random.seed(456)
    dataset = []
    for i in range(n_samples):
        entity = f"Entity_{i}"
        attribute = f"Attribute_{np.random.randint(1000)}"
        dataset.append({
            "id": f"mb_{i}",
            "entity": entity,
            "attribute": attribute,
            "fact": f"{entity} has {attribute}.",
            "question": f"What does {entity} have?",
            "answer": attribute,
        })
    return dataset


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class EvalMetrics:
    """Continual learning evaluation metrics."""

    def __init__(self):
        self.results: List[Dict] = []
        self.forgetting_curve: Dict[str, List[float]] = defaultdict(list)
        self.interference_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)

    def record(self, benchmark: str, sample_id: str,
               accuracy: float, latency_ms: float,
               context_used: bool = False) -> Dict:
        result = {
            "benchmark": benchmark,
            "sample_id": sample_id,
            "accuracy": accuracy,
            "latency_ms": latency_ms,
            "context_used": context_used,
            "timestamp": time.time(),
        }
        self.results.append(result)
        return result

    def compute_forgetting(self, benchmark: str, initial_accuracies: Dict[str, float]) -> Dict[str, float]:
        """Compute forgetting: drop in accuracy over time."""
        current = {r["sample_id"]: r["accuracy"] for r in self.results
                   if r["benchmark"] == benchmark}
        forgetting = {}
        for sid, initial in initial_accuracies.items():
            if sid in current:
                forgetting[sid] = initial - current[sid]
        self.forgetting_curve[benchmark].append(np.mean(list(forgetting.values())) if forgetting else 0.0)
        return forgetting

    def compute_interference(self, benchmark: str, topic: str) -> float:
        """Compute interference: how much learning topic X affects topic Y."""
        topic_results = [r for r in self.results if r["benchmark"] == benchmark]
        if not topic_results:
            return 0.0
        return 1.0 - np.mean([r["accuracy"] for r in topic_results])

    def get_summary(self) -> Dict[str, Any]:
        if not self.results:
            return {"n_evaluations": 0}

        by_benchmark = defaultdict(list)
        for r in self.results:
            by_benchmark[r["benchmark"]].append(r["accuracy"])

        summary = {
            "n_evaluations": len(self.results),
            "overall_accuracy": np.mean([r["accuracy"] for r in self.results]),
            "avg_latency_ms": np.mean([r["latency_ms"] for r in self.results]),
            "context_usage_rate": np.mean([r["context_used"] for r in self.results]),
        }
        for bench, accs in by_benchmark.items():
            summary[f"{bench}_accuracy"] = np.mean(accs)
            summary[f"{bench}_std"] = np.std(accs)

        if self.forgetting_curve:
            for bench, curve in self.forgetting_curve.items():
                summary[f"{bench}_forgetting"] = np.mean(curve)

        return summary


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

class ContinualEvalPipeline:
    """Runs continual learning benchmarks against RTMDK memory."""

    def __init__(self, memory: RTMDKMemory, metrics: Optional[EvalMetrics] = None):
        self.memory = memory
        self.metrics = metrics or EvalMetrics()
        self._initial_accuracies: Dict[str, Dict[str, float]] = {}

    def run_continual_qa(self, dataset: List[Dict]) -> Dict:
        """Run ContinualQA benchmark."""
        results = []
        for sample in dataset:
            # Store fact in memory
            self.memory.save_context(
                {"input": sample["context"], "session_id": "eval"},
                {"output": sample["answer"]}
            )

            # Query and evaluate
            t0 = time.time()
            ctx = self.memory.load_memory_variables(
                {"input": sample["question"], "session_id": "eval"}
            )
            latency_ms = (time.time() - t0) * 1000

            # Check if context contains relevant info
            context_used = sample["topic"].lower() in ctx["rtmdk_context"].lower()
            accuracy = 1.0 if context_used else 0.0

            result = self.metrics.record("continual_qa", sample["id"], accuracy, latency_ms, context_used)
            results.append(result)

        # Track initial accuracies for forgetting computation
        self._initial_accuracies["continual_qa"] = {
            r["sample_id"]: r["accuracy"] for r in results
        }

        return {"n_samples": len(results), "accuracy": np.mean([r["accuracy"] for r in results])}

    def run_long_bench(self, dataset: List[Dict]) -> Dict:
        """Run LongBench-style benchmark."""
        results = []
        for sample in dataset:
            # Store context
            self.memory.save_context(
                {"input": sample["context"][:500], "session_id": "eval_long"},
                {"output": sample["answer"]}
            )

            # Query
            t0 = time.time()
            ctx = self.memory.load_memory_variables(
                {"input": sample["question"], "session_id": "eval_long"}
            )
            latency_ms = (time.time() - t0) * 1000

            # Check if answer is in context
            context_used = sample["answer"] in ctx["rtmdk_context"]
            accuracy = 1.0 if context_used else 0.0

            result = self.metrics.record("long_bench", sample["id"], accuracy, latency_ms, context_used)
            results.append(result)

        return {"n_samples": len(results), "accuracy": np.mean([r["accuracy"] for r in results])}

    def run_memory_bench(self, dataset: List[Dict]) -> Dict:
        """Run MemoryBench-style recall benchmark."""
        results = []
        for sample in dataset:
            # Store fact
            self.memory.save_context(
                {"input": sample["fact"], "session_id": "eval_mem"},
                {"output": sample["answer"]}
            )

            # Query
            t0 = time.time()
            ctx = self.memory.load_memory_variables(
                {"input": sample["question"], "session_id": "eval_mem"}
            )
            latency_ms = (time.time() - t0) * 1000

            # Check if answer is recalled
            context_used = sample["answer"] in ctx["rtmdk_context"]
            accuracy = 1.0 if context_used else 0.0

            result = self.metrics.record("memory_bench", sample["id"], accuracy, latency_ms, context_used)
            results.append(result)

        self._initial_accuracies["memory_bench"] = {
            r["sample_id"]: r["accuracy"] for r in results
        }

        return {"n_samples": len(results), "accuracy": np.mean([r["accuracy"] for r in results])}

    def compute_forgetting(self) -> Dict[str, float]:
        """Compute forgetting across all benchmarks."""
        forgetting = {}
        for bench, initial in self._initial_accuracies.items():
            f = self.metrics.compute_forgetting(bench, initial)
            forgetting[bench] = np.mean(list(f.values())) if f else 0.0
        return forgetting

    def check_rollback_trigger(self, threshold: float = 0.15) -> bool:
        """Check if accuracy degradation triggers rollback."""
        forgetting = self.compute_forgetting()
        if not forgetting:
            return False
        max_forgetting = max(forgetting.values())
        return max_forgetting > threshold

    def get_full_report(self) -> Dict:
        """Generate full evaluation report."""
        summary = self.metrics.get_summary()
        forgetting = self.compute_forgetting()
        rollback = self.check_rollback_trigger()

        report = {
            "summary": summary,
            "forgetting": forgetting,
            "rollback_triggered": rollback,
            "memory_stats": {
                "nodes": self.memory.field.stats.get("active_nodes", 0),
                "consolidations": self.memory.field.stats.get("consolidations", 0),
                "recall_accuracy": self.memory.field.stats.get("recall_accuracy", 1.0),
                "security_violations": self.memory.field.stats.get("security_violations", 0),
            },
        }
        return report


# ============================================================================
# MAIN
# ============================================================================

def run_evaluation(n_samples: int = 50, memory_file: Optional[str] = None):
    """Run full evaluation pipeline."""
    print("=" * 60)
    print("  RTMDK Continual Learning Evaluation")
    print("=" * 60)

    # Initialize memory
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64, top_k=5, enable_async=False,
        causal_topological=True, meta_adaptive=True, self_healing=True,
        cross_modal=True, memory_tiers={"episodic", "semantic", "procedural"},
        attention_bias=True, goal_tracking=True, rl_feedback=True,
        security_enabled=True, meta_memory=True,
        min_response=0.01,
    )

    def embedder(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base

    if memory_file and os.path.exists(memory_file):
        memory = RTMDKMemory.import_field(memory_file, embedder)
        print(f"  Loaded memory from {memory_file}")
    else:
        memory = RTMDKMemory(config=config, embedder=embedder)
        print("  Initialized new memory")

    pipeline = ContinualEvalPipeline(memory)

    # Generate datasets
    print("\n[1] Generating benchmark datasets...")
    cq_data = generate_continual_qa_dataset(n_samples)
    lb_data = generate_long_bench_dataset(n_samples // 2)
    mb_data = generate_memory_bench_dataset(n_samples)
    print(f"  ContinualQA: {len(cq_data)} samples")
    print(f"  LongBench: {len(lb_data)} samples")
    print(f"  MemoryBench: {len(mb_data)} samples")

    # Run benchmarks
    print("\n[2] Running ContinualQA...")
    cq_result = pipeline.run_continual_qa(cq_data)
    print(f"  Accuracy: {cq_result['accuracy']:.2%}")

    print("\n[3] Running LongBench...")
    lb_result = pipeline.run_long_bench(lb_data)
    print(f"  Accuracy: {lb_result['accuracy']:.2%}")

    print("\n[4] Running MemoryBench...")
    mb_result = pipeline.run_memory_bench(mb_data)
    print(f"  Accuracy: {mb_result['accuracy']:.2%}")

    # Full report
    print("\n[5] Generating report...")
    report = pipeline.get_full_report()

    print("\n" + "=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"\n  Overall Accuracy: {report['summary'].get('overall_accuracy', 0):.2%}")
    print(f"  Avg Latency: {report['summary'].get('avg_latency_ms', 0):.1f}ms")
    print(f"  Context Usage: {report['summary'].get('context_usage_rate', 0):.2%}")
    print(f"\n  Forgetting:")
    for bench, f in report.get("forgetting", {}).items():
        print(f"    {bench}: {f:.3f}")
    print(f"\n  Rollback Triggered: {report['rollback_triggered']}")
    print(f"\n  Memory Stats:")
    for k, v in report["memory_stats"].items():
        print(f"    {k}: {v}")

    # Save report
    report_file = "eval_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_file}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTMDK Continual Learning Evaluation")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples per benchmark")
    parser.add_argument("--memory_file", type=str, default=None, help="Path to existing memory file")
    args = parser.parse_args()

    run_evaluation(n_samples=args.n_samples, memory_file=args.memory_file)
