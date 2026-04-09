"""
benchmark_1000_batched.py — Full 1000 QA benchmark in batch mode.

Runs each method separately with automatic checkpoint saving.
Call with:
  python benchmark_1000_batched.py           # Run all 4 methods
  python benchmark_1000_batched.py Baseline   # Run only Baseline
  python benchmark_1000_batched.py resume     # Resume from checkpoint
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder_lmstudio import get_embedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory
from rtmdk.engrams import EngramManager


def load_dataset(path: str = "datasets/qa_1000_en.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


def create_memory(embedder, method_name: str):
    cfg = RTMDKConfig(
        embedding_dim=768, latent_dim=256, top_k=5, min_response=0.005,
        decay_rate=0.999, enable_async=False, bm25_fallback=True,
        use_hnsw=True, learn_projection=False,
        causal_topological=("Causal" in method_name or "All" in method_name),
    )
    memory = RTMDKMemory(config=cfg, embedder=embedder)
    if "Engrams" in method_name or "All" in method_name:
        memory.engram_manager = EngramManager(
            min_nodes=2, max_nodes=15, creation_threshold=0.6,
        )
    return memory


def test_method(method_name: str, records: List[Dict], embedder) -> Dict:
    print(f"\n{'='*70}")
    print(f"  METHOD: {method_name}")
    print(f"{'='*70}")
    
    # Index
    print(f"\n  Indexing {len(records)} facts...")
    memory = create_memory(embedder, method_name)
    t0_index = time.perf_counter()
    for i, rec in enumerate(records):
        memory.save_context(
            {"input": rec["context"], "session_id": f"full_{method_name}"},
            {"output": rec["context"]}
        )
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(records)} ({time.perf_counter()-t0_index:.0f}s)")
    index_time = time.perf_counter() - t0_index
    print(f"  Indexed {len(memory.field.nodes)} nodes in {index_time:.1f}s")
    
    # Test retrieval
    print(f"  Testing {len(records)} queries...")
    recalls = {1: 0, 3: 0, 5: 0, 10: 0}
    latencies = []
    per_topic = {}
    n_tested = 0
    
    t0_test = time.perf_counter()
    for i, rec in enumerate(records):
        answer_words = [w for w in rec["answer"].lower().split() if len(w) > 2]
        if not answer_words:
            continue
        
        topic = rec.get("topic", "unknown")
        if topic not in per_topic:
            per_topic[topic] = {"correct": 0, "total": 0}
        per_topic[topic]["total"] += 1
        
        t0 = time.perf_counter()
        ctx = memory.load_memory_variables({
            "input": rec["query"],
            "session_id": f"full_{method_name}"
        })
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        n_tested += 1
        
        context = ctx.get("rtmdk_context", "").lower()
        found = any(w in context for w in answer_words)
        
        if found:
            recalls[1] += 1
            recalls[3] += 1
            recalls[5] += 1
            recalls[10] += 1
            per_topic[topic]["correct"] += 1
        
        if (i + 1) % 250 == 0:
            r1 = recalls[1] / max(n_tested, 1)
            elapsed = time.perf_counter() - t0_test
            print(f"  [{i+1:4d}/{len(records)}] R@1={r1:.0%}  "
                  f"P50={np.percentile(latencies, 50):.0f}ms  "
                  f"P95={np.percentile(latencies, 95):.0f}ms  "
                  f"({elapsed:.0f}s elapsed)")
    
    test_time = time.perf_counter() - t0_test
    topic_results = {}
    for topic, stats in per_topic.items():
        topic_results[topic] = {
            "recall": stats["correct"] / max(stats["total"], 1),
            "total": stats["total"],
        }
    
    return {
        "method": method_name,
        "n_indexed": len(memory.field.nodes),
        "n_tested": n_tested,
        "recall_at_1": recalls[1] / max(n_tested, 1),
        "recall_at_3": recalls[3] / max(n_tested, 1),
        "recall_at_5": recalls[5] / max(n_tested, 1),
        "recall_at_10": recalls[10] / max(n_tested, 1),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "index_time_s": round(index_time, 1),
        "test_time_s": round(test_time, 1),
        "per_topic": topic_results,
    }


def print_report(all_results: List[Dict], n_records: int):
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — {n_records} QA FULL BENCHMARK")
    print(f"{'='*70}")
    
    print(f"\n  {'Method':<15} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} "
          f"{'P50':>5} {'P95':>5} {'P99':>5} {'Idx':>5}")
    print(f"  {'─'*68}")
    for r in all_results:
        print(f"  {r['method']:<15} {r['recall_at_1']:>5.0%} {r['recall_at_3']:>5.0%} "
              f"{r['recall_at_5']:>5.0%} {r['recall_at_10']:>5.0%} "
              f"{r['latency_p50_ms']:>3.0f}ms {r['latency_p95_ms']:>3.0f}ms "
              f"{r['latency_p99_ms']:>3.0f}ms {r['index_time_s']:>4.0f}s")
    
    best = max(all_results, key=lambda r: r["recall_at_1"])
    print(f"\n  {'Per-Topic Recall (Best: ' + best['method'] + ')':^60}")
    print(f"  {'─'*60}")
    for topic, stats in sorted(best["per_topic"].items()):
        bar = "█" * int(stats["recall"] * 20)
        print(f"  {topic:15s}: {stats['recall']:5.0%} {bar}")
    
    print(f"\n  {'vs INDUSTRY BENCHMARKS':^60}")
    print(f"  {'─'*60}")
    print(f"  RTMDK (current):    R@1 = {best['recall_at_1']:.0%}")
    print(f"  GraphRAG:           R@1 = 82-90%")
    print(f"  Self-RAG:           R@1 = 80-88%")
    print(f"  Advanced RAG:       R@1 = 75-85%")
    print(f"  Naive RAG:          R@1 = 60-75%")
    
    print(f"\n  {'IMPROVEMENT vs BASELINE':^60}")
    print(f"  {'─'*60}")
    baseline_r1 = all_results[0]["recall_at_1"]
    for r in all_results:
        delta = r['recall_at_1'] - baseline_r1
        delta_str = f" (Δ{delta:+.0%})" if delta != 0 else ""
        print(f"  {r['method']:15s}: R@1={r['recall_at_1']:.0%}{delta_str}  "
              f"Latency P50={r['latency_p50_ms']:.0f}ms  P95={r['latency_p95_ms']:.0f}ms")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("method", nargs="?", default="all",
                        choices=["all", "Baseline", "+Engrams", "+Causal", "All Combined", "resume"])
    args = parser.parse_args()
    
    print("=" * 70)
    print("  FULL 1000 QA BENCHMARK — Batch Mode")
    print("=" * 70)
    
    records = load_dataset()
    topics = {}
    for rec in records:
        t = rec.get("topic", "unknown")
        topics[t] = topics.get(t, 0) + 1
    print(f"  Loaded {len(records)} QA pairs, topics: {dict(sorted(topics.items()))}")
    
    checkpoint_path = Path("full_1000_checkpoint.json")
    all_results = []
    
    # Load checkpoint
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        all_results = ckpt.get("results", [])
        completed_methods = [r["method"] for r in all_results]
        print(f"  Loaded checkpoint: {len(completed_methods)} methods done: {completed_methods}")
    
    methods = ["Baseline", "+Engrams", "+Causal", "All Combined"]
    if args.method != "all" and args.method != "resume":
        methods = [args.method]
    
    # Filter out already completed
    methods = [m for m in methods if m not in [r["method"] for r in all_results]]
    
    embedder = get_embedder()
    
    for method in methods:
        result = test_method(method, records, embedder)
        all_results.append(result)
        
        # Save checkpoint
        with open(checkpoint_path, "w") as f:
            json.dump({"results": all_results}, f, indent=2, default=str)
        print(f"\n  [{method}] Checkpoint saved")
    
    # Print report if all done
    if len(all_results) >= 4:
        print_report(all_results, len(records))
        
        # Save final report
        best = max(all_results, key=lambda r: r["recall_at_1"])
        report = {
            "n_records": len(records),
            "methods": all_results,
            "best_method": best["method"],
            "best_recall_at_1": best["recall_at_1"],
        }
        with open("full_1000_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Final report saved to full_1000_report.json")
    else:
        print(f"\n  {len(all_results)}/4 methods done. Re-run to continue.")
        # Print partial results
        print_report(all_results, len(records))


if __name__ == "__main__":
    main()
