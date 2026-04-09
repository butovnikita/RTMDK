"""
benchmark_rag_baseline.py — FAISS+RAG Baseline for comparison with RTMDK.

Measures the same metrics as benchmark_rtmdk.py but uses simple FAISS-like
vector search (numpy-based, no external deps) with naive chunk-based retrieval.

Usage:
    python benchmark_rag_baseline.py [--n_dialogues 500] [--report rag_baseline_report.json]
"""

import os
import sys
import json
import time
import argparse
import random
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_rtmdk import generate_dialogue_dataset


# ============================================================================
# SIMPLE FAISS-LIKE RETRIEVAL (numpy-based, no external deps)
# ============================================================================

class FAISSBaseline:
    """Simple FAISS-like retrieval baseline using numpy dot product."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.documents: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.session_map: Dict[str, List[int]] = {}

    @staticmethod
    def _embed(text: str, dim: int = 768) -> np.ndarray:
        np.random.seed(42)
        base = np.random.randn(dim).astype(np.float32) * 0.01
        tokens = text.lower().split()
        for tok in tokens[:20]:
            np.random.seed(hash(tok + "benchmark_seed") % 2**32)
            direction = np.random.randn(dim).astype(np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            base += direction * 0.5
        return base

    def add_document(self, doc_id: str, text: str, session_id: str = "default"):
        idx = len(self.documents)
        self.documents.append(text)
        emb = self._embed(text)
        self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])
        if session_id not in self.session_map:
            self.session_map[session_id] = []
        self.session_map[session_id].append(idx)

    def retrieve(self, query: str, session_id: str = "default", top_k: int = 5) -> List[Dict]:
        query_emb = self._embed(query)

        # Session-aware retrieval
        if session_id in self.session_map:
            indices = self.session_map[session_id]
            session_embs = self.embeddings[indices]
            scores = session_embs @ query_emb
            top_local = np.argsort(scores)[::-1][:top_k]
            results = [(indices[i], float(scores[i])) for i in top_local]
        else:
            # Global fallback
            scores = self.embeddings @ query_emb
            top_global = np.argsort(scores)[::-1][:top_k]
            results = [(int(i), float(scores[i])) for i in top_global]

        return [
            {"doc_id": idx, "text": self.documents[idx], "score": score}
            for idx, score in results
        ]


# ============================================================================
# BENCHMARK ENGINE
# ============================================================================

class RAGBaselineBenchmark:
    """Runs FAISS baseline benchmark."""

    def __init__(self):
        self.retriever = FAISSBaseline()
        self.latencies: List[float] = []

    def store_facts(self, dataset: List[Dict]):
        for item in dataset:
            self.retriever.add_document(
                item["id"], item["fact"], item["session_id"]
            )

    def run_queries(self, dataset: List[Dict], top_k: int = 5) -> Dict[str, Any]:
        n_correct = 0
        total_tokens_in = 0
        total_tokens_out = 0
        query_results = []

        for item in dataset:
            t0 = time.perf_counter()
            results = self.retriever.retrieve(item["query"], item["session_id"], top_k)
            latency_ms = (time.perf_counter() - t0) * 1000
            self.latencies.append(latency_ms)

            context = " ".join(r["text"] for r in results)
            answer_lower = item["expected_answer"].lower()
            context_lower = context.lower()
            answer_keywords = [w for w in answer_lower.split() if len(w) > 3]
            matches = sum(1 for kw in answer_keywords if kw in context_lower)
            recall = matches / len(answer_keywords) if answer_keywords else 0.0

            if recall >= 0.5:
                n_correct += 1

            tokens_in = len(item["query"].split())
            tokens_out = len(context.split())
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out

            query_results.append({
                "id": item["id"],
                "recall": recall,
                "latency_ms": latency_ms,
            })

        n_queries = len(dataset)
        latencies_sorted = sorted(self.latencies)

        return {
            "n_queries": n_queries,
            "n_stored_facts": len(self.retriever.documents),
            "context_recall": n_correct / max(n_queries, 1),
            "token_efficiency": total_tokens_in / max(total_tokens_out, 1),
            "latency_p50_ms": latencies_sorted[len(latencies_sorted) // 2] if latencies_sorted else 0,
            "latency_p95_ms": latencies_sorted[int(len(latencies_sorted) * 0.95)] if latencies_sorted else 0,
            "latency_p99_ms": latencies_sorted[int(len(latencies_sorted) * 0.99)] if latencies_sorted else 0,
            "latency_mean_ms": np.mean(self.latencies) if self.latencies else 0,
        }


# ============================================================================
# MAIN
# ============================================================================

def run_baseline_benchmark(n_dialogues: int = 500, report_path: str = "rag_baseline_report.json"):
    print("=" * 60)
    print("  FAISS+RAG Baseline Benchmark")
    print("=" * 60)

    print(f"\n[1] Generating dataset ({n_dialogues} dialogues)...")
    dataset = generate_dialogue_dataset(n_dialogues)

    bench = RAGBaselineBenchmark()

    print(f"\n[2] Storing {len(dataset)} facts...")
    bench.store_facts(dataset)

    print(f"\n[3] Running {len(dataset)} queries...")
    report = bench.run_queries(dataset)

    print("\n" + "=" * 60)
    print("  BASELINE RESULTS")
    print("=" * 60)
    print(f"  Documents:             {report['n_stored_facts']}")
    print(f"  Context recall:        {report['context_recall']:.2%}")
    print(f"  Token efficiency:      {report['token_efficiency']:.2f}")
    print(f"  Latency p50:           {report['latency_p50_ms']:.2f}ms")
    print(f"  Latency p95:           {report['latency_p95_ms']:.2f}ms")
    print(f"  Latency p99:           {report['latency_p99_ms']:.2f}ms")
    print(f"  Latency mean:          {report['latency_mean_ms']:.2f}ms")
    print("=" * 60)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {report_path}")

    return report


def compare_reports(rtmdk_path: str, rag_path: str):
    """Compare RTMDK and RAG baseline reports."""
    with open(rtmdk_path) as f:
        rtmdk = json.load(f)
    with open(rag_path) as f:
        rag = json.load(f)

    print("\n" + "=" * 60)
    print("  COMPARISON: RTMDK vs FAISS+RAG Baseline")
    print("=" * 60)

    for metric in ["context_recall", "token_efficiency", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]:
        r_val = rtmdk.get(metric, 0)
        b_val = rag.get(metric, 0)
        if "latency" in metric:
            diff = ((b_val - r_val) / max(b_val, 0.01)) * 100
            print(f"  {metric:25s}: RTMDK={r_val:8.2f}  Baseline={b_val:8.2f}  Delta={diff:+6.1f}%")
        else:
            diff = ((r_val - b_val) / max(b_val, 0.01)) * 100
            print(f"  {metric:25s}: RTMDK={r_val:8.4f}  Baseline={b_val:8.4f}  Delta={diff:+6.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FAISS+RAG Baseline Benchmark")
    parser.add_argument("--n_dialogues", type=int, default=500, help="Number of dialogue pairs")
    parser.add_argument("--report", type=str, default="rag_baseline_report.json", help="Output report path")
    parser.add_argument("--compare_with", type=str, default=None, help="RTMDK report path to compare with")
    args = parser.parse_args()

    report = run_baseline_benchmark(n_dialogues=args.n_dialogues, report_path=args.report)

    if args.compare_with:
        compare_reports(args.compare_with, args.report)
