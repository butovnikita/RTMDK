"""End-to-end quality benchmarks for working features.

Tests:
- Quantization recall degradation on synthetic nearest-neighbor retrieval
- SentenceReranker NDCG improvement on mock results
- CascadeRouter routing accuracy on real queries
- QueryRewriter expansion rate on dataset
- QueryIntentClassifier accuracy on expanded test set
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_dataset(name: str):
    path = PROJECT_ROOT / "datasets" / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("records", data.get("data", list(data.values())))
    return data


def eval_quantization_recall() -> Dict:
    """Measure recall degradation from fp32 -> fp16 -> int8."""
    print("\n=== Quantization Recall Impact ===")
    try:
        from rtmdk.memory.quantization import QuantizationHelper

        rng = np.random.default_rng(42)
        n_queries = 100
        n_docs = 1000
        dim = 384

        # Generate random embeddings (normalized)
        docs = rng.standard_normal((n_docs, dim)).astype(np.float32)
        docs = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-8)
        queries = rng.standard_normal((n_queries, dim)).astype(np.float32)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)

        # Ground truth: fp32 top-10
        gt = queries @ docs.T  # (n_queries, n_docs)
        gt_top10 = np.argsort(-gt, axis=1)[:, :10]

        def _recall(q_emb, top_k=10):
            scores = q_emb @ docs.T
            pred_top = np.argsort(-scores, axis=1)[:, :top_k]
            hits = sum(
                len(set(pred_top[i]) & set(gt_top10[i]))
                for i in range(n_queries)
            )
            return hits / (n_queries * top_k)

        # fp16
        qh_fp16 = QuantizationHelper("fp16")
        docs_fp16 = qh_fp16.quantize(docs)
        # int8
        qh_int8 = QuantizationHelper("int8")
        docs_int8_tuple = qh_int8.quantize(docs)
        if isinstance(docs_int8_tuple, tuple):
            docs_int8, scale, zp = docs_int8_tuple
            docs_int8_dq = qh_int8.dequantize(docs_int8, scale, zp)
        else:
            docs_int8_dq = docs_int8_tuple.astype(np.float32)

        r_fp32 = _recall(docs)
        r_fp16 = _recall(docs_fp16.astype(np.float32))
        r_int8 = _recall(docs_int8_dq)

        result = {
            "feature": "QuantizationRecall",
            "recall@10_fp32": round(r_fp32, 4),
            "recall@10_fp16": round(r_fp16, 4),
            "recall@10_int8": round(r_int8, 4),
            "fp16_degradation_pct": round((1 - r_fp16 / r_fp32) * 100, 2) if r_fp32 else 0,
            "int8_degradation_pct": round((1 - r_int8 / r_fp32) * 100, 2) if r_fp32 else 0,
            "status": "PASS" if r_int8 >= r_fp32 * 0.95 else "FAIL",
            "note": "Degradation <5% is acceptable for production",
        }
    except Exception as e:
        result = {"feature": "QuantizationRecall", "status": "ERROR", "note": str(e)}
    print(json.dumps(result, indent=2))
    return result


def eval_reranker_quality() -> Dict:
    """Measure NDCG improvement from sentence-level reranking."""
    print("\n=== SentenceReranker Quality ===")
    try:
        from rtmdk.memory.rag_quality import SentenceReranker

        # Mock embedder: simple keyword overlap -> embedding
        vocab = {"france": 0, "paris": 1, "capital": 2, "germany": 3,
                 "berlin": 4, "spain": 5, "madrid": 6, "europe": 7}
        dim = len(vocab)

        def mock_embed(text: str):
            emb = np.zeros(dim, dtype=np.float32)
            for w in text.lower().split():
                if w in vocab:
                    emb[vocab[w]] = 1.0
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-8) if norm else emb

        reranker = SentenceReranker(embedder=mock_embed, batch_size=4)

        # Mock nodes: one highly relevant, one partially relevant, one irrelevant
        class MockNode:
            def __init__(self, text):
                self.content = {"text": text}

        results = [
            ("n1", 0.5, MockNode("France is a country in Europe. Its capital is Paris.")),
            ("n2", 0.6, MockNode("Germany and Berlin are important. Europe has many capitals.")),
            ("n3", 0.4, MockNode("Spain is warm. Madrid has museums.")),
        ]

        reranked = reranker.rerank("What is the capital of France?", results, top_k=3)
        ids = [r[0] for r in reranked]

        # Expected: n1 should be first because it contains "capital" and "france" and "paris"
        result = {
            "feature": "SentenceRerankerQuality",
            "top1": ids[0],
            "top3": ids,
            "status": "PASS" if ids[0] == "n1" else "FAIL",
            "note": "n1 should win due to sentence-level match",
        }
    except Exception as e:
        result = {"feature": "SentenceRerankerQuality", "status": "ERROR", "note": str(e)}
    print(json.dumps(result, indent=2))
    return result


def eval_cascade_router_accuracy() -> Dict:
    """Test cascade router on real queries."""
    print("\n=== CascadeRouter Accuracy ===")
    try:
        from rtmdk.production.cascade_router import AdaptiveCascadeRouter as CascadeRouter

        router = CascadeRouter()
        dataset = _load_dataset("comprehensive_500")
        queries = [item.get("query", "") for item in dataset[:100] if item.get("query")]

        routed = {"fast": 0, "standard": 0, "deep": 0}
        for q in queries:
            route = router.route(q)
            routed[route] = routed.get(route, 0) + 1

        # Most queries in comprehensive_500 are factual -> fast
        fast_pct = routed.get("fast", 0) / len(queries) * 100 if queries else 0

        result = {
            "feature": "CascadeRouterAccuracy",
            "total_queries": len(queries),
            "routed": routed,
            "fast_pct": round(fast_pct, 1),
            "status": "PASS" if fast_pct >= 50 else "FAIL",
            "note": "Expect mostly fast (factual) on QA dataset",
        }
    except Exception as e:
        result = {"feature": "CascadeRouterAccuracy", "status": "ERROR", "note": str(e)}
    print(json.dumps(result, indent=2))
    return result


def eval_query_rewriter_expansion() -> Dict:
    """Measure how often QueryRewriter expands queries."""
    print("\n=== QueryRewriter Expansion Rate ===")
    try:
        from rtmdk.memory.explainability import QueryRewriter

        # Mock embedder for keyword overlap heuristic
        vocab = {"france": 0, "paris": 1, "capital": 2, "european": 3, "city": 4}
        dim = len(vocab)
        def mock_embed(text: str):
            emb = np.zeros(dim, dtype=np.float32)
            for w in text.lower().split():
                if w in vocab:
                    emb[vocab[w]] = 1.0
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-8) if norm else emb

        rewriter = QueryRewriter(embedder=mock_embed)
        dataset = _load_dataset("comprehensive_500")
        queries = [item.get("query", "") for item in dataset[:200] if item.get("query")]

        expanded = 0
        unchanged = 0
        class MockNode:
            def __init__(self, text):
                self.content = {"text": text}
        mock_results = [("n1", 0.2, MockNode("Paris is the capital of France and a major European city."))]
        for q in queries:
            new_q = rewriter.rewrite(q, mock_results)
            if new_q != q:
                expanded += 1
            else:
                unchanged += 1

        rate = expanded / len(queries) if queries else 0.0
        result = {
            "feature": "QueryRewriterExpansion",
            "total": len(queries),
            "expanded": expanded,
            "unchanged": unchanged,
            "expansion_rate": round(rate, 3),
            "status": "PASS" if rate >= 0.05 else "FAIL",
            "note": "Low rate is OK if expansions are high-quality",
        }
    except Exception as e:
        result = {"feature": "QueryRewriterExpansion", "status": "ERROR", "note": str(e)}
    print(json.dumps(result, indent=2))
    return result


def eval_intent_classifier_expanded() -> Dict:
    """Test intent classifier on expanded set."""
    print("\n=== QueryIntentClassifier Expanded ===")
    try:
        from rtmdk.memory.explainability import QueryIntentClassifier

        clf = QueryIntentClassifier()
        cases = [
            ("What is the capital of France?", "factual"),
            ("How does photosynthesis work?", "factual"),
            ("Tell me about black holes.", "exploratory"),
            ("What are the latest trends in AI?", "exploratory"),
            ("Hello, how are you?", "conversational"),
            ("Thanks for the help!", "conversational"),
            ("Who wrote Pride and Prejudice?", "factual"),
            ("Compare Python and Java.", "comparative"),
            ("Can you explain quantum mechanics?", "factual"),
            ("Good morning!", "conversational"),
        ]

        correct = sum(1 for q, expected in cases if clf.classify(q) == expected)
        acc = correct / len(cases)

        result = {
            "feature": "QueryIntentClassifierExpanded",
            "accuracy": round(acc, 2),
            "correct": correct,
            "total": len(cases),
            "status": "PASS" if acc >= 0.8 else "FAIL",
            "note": f"{correct}/{len(cases)} correct",
        }
    except Exception as e:
        result = {"feature": "QueryIntentClassifierExpanded", "status": "ERROR", "note": str(e)}
    print(json.dumps(result, indent=2))
    return result


def eval_query_decomposer_synthetic() -> Dict:
    """Test decomposer on synthetic multi-hop queries."""
    print("\n=== QueryDecomposer Synthetic ===")
    try:
        from rtmdk.memory.rag_quality import QueryDecomposer

        dec = QueryDecomposer(llm_client=None)
        cases = [
            ("What is the capital of France and who is its president?",
             ["What is the capital of France", "who is its president"]),
            ("Compare Python to Java.",
             ["Compare Python", "Java"]),
            ("What causes earthquakes and how do they affect buildings?",
             ["What causes earthquakes", "how do they affect buildings"]),
            ("Who wrote Pride and Prejudice?",
             ["Who wrote Pride and Prejudice?"]),  # named entity, should NOT split
            ("What is the speed of light?",
             ["What is the speed of light?"]),  # single-hop
        ]

        correct = 0
        for q, expected in cases:
            got = dec.decompose(q)
            # Allow fuzzy match: each expected piece should be present
            if len(got) == len(expected):
                if all(any(exp in g or g in exp for g in got) for exp in expected):
                    correct += 1

        acc = correct / len(cases)
        result = {
            "feature": "QueryDecomposerSynthetic",
            "accuracy": round(acc, 2),
            "correct": correct,
            "total": len(cases),
            "status": "PASS" if acc >= 0.6 else "FAIL",
            "note": f"{correct}/{len(cases)} correct on multi-hop queries",
        }
    except Exception as e:
        result = {"feature": "QueryDecomposerSynthetic", "status": "ERROR", "note": str(e)}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    results = [
        eval_quantization_recall(),
        eval_reranker_quality(),
        eval_cascade_router_accuracy(),
        eval_query_rewriter_expansion(),
        eval_intent_classifier_expanded(),
        eval_query_decomposer_synthetic(),
    ]
    passed = sum(1 for r in results if r.get("status") == "PASS")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    errors = sum(1 for r in results if r.get("status") == "ERROR")
    print(f"\n=== SUMMARY ===")
    print(f"PASS: {passed}, FAIL: {failed}, ERROR: {errors}")
    out = PROJECT_ROOT / "scripts" / "eval_working_features_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out}")
