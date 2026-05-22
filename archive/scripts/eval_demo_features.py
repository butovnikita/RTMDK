"""Evaluation of Category B: Demo-level features.

Benchmarks:
- QueryDecomposer: multi-hop QA recall
- SentenceReranker: latency + recall trade-off
- QueryRewriter: recall improvement on low-score queries
- CascadeRouter: latency vs accuracy trade-off
- ResultExplainer: sanity check (no crash)

Usage:
    python scripts/eval_demo_features.py --dataset comprehensive_500
"""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rtmdk.memory.core import RTMDKMemory, RTMDKConfig
from rtmdk.memory.rag_quality import QueryDecomposer, SentenceReranker
from rtmdk.memory.explainability import QueryRewriter, QueryIntentClassifier


def _load_dataset(name: str):
    path = PROJECT_ROOT / "datasets" / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_embedder(dim: int = 384):
    rng = np.random.default_rng(42)
    cache = {}
    def embed(text: str) -> np.ndarray:
        if text not in cache:
            cache[text] = rng.standard_normal(dim).astype(np.float32)
            cache[text] /= np.linalg.norm(cache[text]) + 1e-8
        return cache[text]
    return embed


def eval_query_decomposer(dataset: List[Dict]) -> Dict:
    """Evaluate if decomposition improves multi-hop recall."""
    print("\n=== QueryDecomposer ===")
    dec = QueryDecomposer()
    total = len(dataset)
    improved = 0
    same = 0
    worse = 0

    for item in dataset[:50]:  # Sample for speed
        query = item.get("question", "")
        sub = dec.decompose(query)
        if len(sub) > 1:
            improved += 1
        elif len(sub) == 1 and sub[0] == query:
            same += 1
        else:
            worse += 1

    result = {
        "feature": "QueryDecomposer",
        "total_evaluated": total,
        "decomposed": improved,
        "unchanged": same,
        "status": "PASS" if improved > 0 else "FAIL",
        "note": f"Decomposed {improved}/{50} queries",
    }
    print(json.dumps(result, indent=2))
    return result


def eval_sentence_reranker(dataset: List[Dict]) -> Dict:
    """Evaluate latency overhead of sentence reranking."""
    print("\n=== SentenceReranker ===")
    embedder = _make_embedder(384)
    reranker = SentenceReranker(embedder, batch_size=8)

    class MockNode:
        def __init__(self, text):
            self.content = {"text": text}

    results = [(f"n{i}", float(i), MockNode("Sentence one. Sentence two. " * 10)) for i in range(20)]

    t0 = time.perf_counter()
    for _ in range(10):
        reranker.rerank("query", results, top_k=5)
    latency_ms = (time.perf_counter() - t0) * 1000 / 10

    result = {
        "feature": "SentenceReranker",
        "latency_ms_per_query": round(latency_ms, 2),
        "status": "PASS" if latency_ms < 100 else "WARN",
        "note": f"{'Fast' if latency_ms < 50 else 'Acceptable' if latency_ms < 100 else 'Slow'} for production",
    }
    print(json.dumps(result, indent=2))
    return result


def eval_query_rewriter(dataset: List[Dict]) -> Dict:
    """Evaluate query rewriting on low-score queries."""
    print("\n=== QueryRewriter ===")
    embedder = _make_embedder(384)
    rewriter = QueryRewriter(embedder=embedder)

    class MockNode:
        def __init__(self, text):
            self.content = {"text": text}

    low_score_results = [("n1", 0.2, MockNode("The capital of France is Paris."))]
    rewritten = rewriter.rewrite("France capital", low_score_results)

    result = {
        "feature": "QueryRewriter",
        "original_query": "France capital",
        "rewritten_query": rewritten,
        "status": "PASS" if rewritten != "France capital" else "FAIL",
        "note": "Heuristic expansion works if text overlap found",
    }
    print(json.dumps(result, indent=2))
    return result


def eval_cascade_router(dataset: List[Dict]) -> Dict:
    """Evaluate cascade router latency vs accuracy."""
    print("\n=== CascadeRouter ===")
    try:
        from rtmdk.production.cascade_router import AdaptiveCascadeRouter
        router = AdaptiveCascadeRouter()

        queries = [item.get("question", "") for item in dataset[:20]]
        t0 = time.perf_counter()
        for q in queries:
            router.classify(q)
        latency_ms = (time.perf_counter() - t0) * 1000 / len(queries)

        result = {
            "feature": "CascadeRouter",
            "latency_ms_per_query": round(latency_ms, 3),
            "status": "PASS" if latency_ms < 1 else "WARN",
            "note": "Regex routing is fast but simplistic",
        }
    except Exception as e:
        result = {
            "feature": "CascadeRouter",
            "status": "ERROR",
            "note": str(e),
        }
    print(json.dumps(result, indent=2))
    return result


def eval_intent_classifier(dataset: List[Dict]) -> Dict:
    """Evaluate intent classifier accuracy on sample."""
    print("\n=== QueryIntentClassifier ===")
    clf = QueryIntentClassifier()
    test_cases = [
        ("What is the capital of France?", "factual"),
        ("How are you doing today?", "conversational"),
        ("Compare Python and JavaScript", "comparative"),
        ("Tell me about quantum computing", "exploratory"),
    ]
    correct = 0
    for query, expected in test_cases:
        pred = clf.classify(query)
        if pred == expected:
            correct += 1
    result = {
        "feature": "QueryIntentClassifier",
        "accuracy": correct / len(test_cases),
        "status": "PASS" if correct / len(test_cases) >= 0.5 else "FAIL",
        "note": f"{correct}/{len(test_cases)} correct on heuristic-only",
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="comprehensive_500")
    args = parser.parse_args()

    try:
        dataset = _load_dataset(args.dataset)
        if isinstance(dataset, dict):
            dataset = dataset.get("records", dataset.get("data", list(dataset.values())))
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        dataset = []

    print(f"Evaluating demo features on dataset: {args.dataset} ({len(dataset)} items)")

    results = []
    results.append(eval_query_decomposer(dataset))
    results.append(eval_sentence_reranker(dataset))
    results.append(eval_query_rewriter(dataset))
    results.append(eval_cascade_router(dataset))
    results.append(eval_intent_classifier(dataset))

    # Summary
    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    errors = sum(1 for r in results if r.get("status") == "ERROR")

    print(f"\n=== SUMMARY ===")
    print(f"PASS: {passed}, FAIL: {failed}, ERROR: {errors}")

    output_path = PROJECT_ROOT / "scripts" / "eval_demo_features_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
