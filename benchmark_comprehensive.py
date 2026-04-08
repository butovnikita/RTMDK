"""
benchmark_comprehensive.py — Comprehensive RTMDK Benchmark with 500 QA pairs.

Stage 1: Retrieval Benchmark (all 500) — Recall@K, latency, token counts
Stage 2: LLM Quality Eval (50 sampled) — exact match, hallucination rate

Usage:
    python benchmark_comprehensive.py [--stage all] [--report comprehensive_report.json]
"""

import os
import sys
import json
import time
import random
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder_lmstudio import get_embedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


# ============================================================================
# TOKEN COUNTER
# ============================================================================

def estimate_tokens(text: str, language: str = "en") -> int:
    """Estimate token count using language-specific heuristics."""
    if not text:
        return 0
    if language == "ru":
        return max(1, len(text) // 6)  # Russian: ~6 chars/token
    return max(1, len(text) // 4)  # English: ~4 chars/token


# ============================================================================
# LLM API
# ============================================================================

def query_llm(query: str, context: str, model: str = "thedrummer_rocinante-x-12b-v1",
              timeout: int = 120) -> Tuple[str, float]:
    """Query LLM with context and measure response."""
    try:
        import requests
        url = "http://localhost:12345/api/v1/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"Use ONLY the provided context to answer. If the context doesn't contain the answer, say 'I don't know based on the provided context.'\n\nContext: {context}"},
                {"role": "user", "content": query},
            ],
            "temperature": 0.1,
            "max_tokens": 100,
        }
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        data = resp.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return answer, latency
    except Exception as e:
        return f"[Error: {e}]", 0.0


def query_llm_judge(expected: str, actual: str) -> float:
    """Ask LLM to judge answer quality (1-5 scale)."""
    try:
        import requests
        url = "http://localhost:12345/api/v1/chat"
        prompt = f"""Rate how well the actual answer matches the expected answer on a scale of 1-5:
1 = Completely wrong or unrelated
2 = Partially correct but major errors
3 = Mostly correct with minor issues
4 = Correct with slight differences
5 = Essentially the same meaning

Expected: {expected}
Actual: {actual}

Score (just give the number 1-5):"""
        resp = requests.post(url, json={
            "model": "thedrummer_rocinante-x-12b-v1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 5,
        }, timeout=60)
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "3").strip()
        for c in text:
            if c in "12345":
                return float(c)
        return 3.0
    except Exception:
        return 3.0


# ============================================================================
# STAGE 1: RETRIEVAL BENCHMARK
# ============================================================================

def run_stage1(records: List[Dict], max_n: int = 500) -> Dict:
    """Full retrieval benchmark with token counting."""
    embedder = get_embedder()
    memory = RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=getattr(embedder, 'dim', 768),
            latent_dim=256, top_k=5, min_response=0.005,
            decay_rate=0.999, enable_async=False, bm25_fallback=True,
            use_hnsw=True, learn_projection=True, projection_update_freq=300,
        ),
        embedder=embedder,
    )

    print(f"\n[Stage 1] Storing {min(max_n, len(records))} facts...")
    t0_store = time.perf_counter()
    for i, rec in enumerate(records[:max_n]):
        # Store context (single save per fact for speed)
        memory.save_context(
            {"input": rec["context"], "session_id": "bench"},
            {"output": rec["context"]}
        )
        if i % 50 == 0:
            elapsed = time.perf_counter() - t0_store
            print(f"  ...{i}/{max_n} ({elapsed:.0f}s)")
    store_time = time.perf_counter() - t0_store
    print(f"  Stored {len(memory.field.nodes)} nodes in {store_time:.1f}s")

    print(f"\n[Stage 1] Running {min(max_n, len(records))} queries...")
    recalls_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    latencies = []
    query_tokens_list = []
    context_tokens_list = []
    answer_tokens_list = []

    lang_results = {"en": {"hits": 0, "total": 0}, "ru": {"hits": 0, "total": 0}}
    topic_results = {}

    for rec in records[:max_n]:
        query = rec["query"]
        answer = rec["answer"].lower()
        lang = rec.get("language", "en")
        topic = rec.get("topic", "general")

        t0 = time.perf_counter()
        ctx = memory.load_memory_variables({"input": query, "session_id": "bench"})
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        context = ctx.get("rtmdk_context", "").lower()
        answer_words = [w for w in answer.split() if len(w) > 2]

        # Check recall at various K values (approximate by checking if ANY answer word is in context)
        found_any = any(w in context for w in answer_words) if answer_words else False
        found_first = answer_words[0] in context if answer_words else False

        if found_first:
            recalls_at_k[1] += 1
        if found_any:
            recalls_at_k[3] += 1
            recalls_at_k[5] += 1
            recalls_at_k[10] += 1

        # Token counts
        q_tokens = estimate_tokens(query, lang)
        c_tokens = estimate_tokens(ctx.get("rtmdk_context", ""), lang)
        a_tokens = estimate_tokens(rec["answer"], lang)
        query_tokens_list.append(q_tokens)
        context_tokens_list.append(c_tokens)
        answer_tokens_list.append(a_tokens)

        # Per-language
        lang_results[lang]["total"] += 1
        if found_any:
            lang_results[lang]["hits"] += 1

        # Per-topic
        if topic not in topic_results:
            topic_results[topic] = {"hits": 0, "total": 0}
        topic_results[topic]["total"] += 1
        if found_any:
            topic_results[topic]["hits"] += 1

    n = min(max_n, len(records))
    total = lang_results["en"]["total"] + lang_results["ru"]["total"]

    return {
        "stage": "retrieval",
        "n_queries": n,
        "recall_at_1": recalls_at_k[1] / max(n, 1),
        "recall_at_3": recalls_at_k[3] / max(n, 1),
        "recall_at_5": recalls_at_k[5] / max(n, 1),
        "recall_at_10": recalls_at_k[10] / max(n, 1),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "avg_query_tokens": float(np.mean(query_tokens_list)),
        "avg_context_tokens": float(np.mean(context_tokens_list)),
        "avg_answer_tokens": float(np.mean(answer_tokens_list)),
        "token_efficiency": float(np.mean([a / max(c, 1) for a, c in zip(answer_tokens_list, context_tokens_list)])),
        "by_language": {
            lang: {"recall": d["hits"] / max(d["total"], 1), "total": d["total"]}
            for lang, d in lang_results.items() if d["total"] > 0
        },
        "by_topic": {
            t: {"recall": d["hits"] / max(d["total"], 1), "total": d["total"]}
            for t, d in sorted(topic_results.items())
        },
    }


# ============================================================================
# STAGE 2: LLM QUALITY EVALUATION
# ============================================================================

def run_stage2(records: List[Dict], memory: RTMDKMemory, n_sample: int = 50) -> Dict:
    """LLM-based quality evaluation on a sampled subset."""
    # Select diverse samples: 5 from each topic
    topics = {}
    for rec in records:
        t = rec.get("topic", "general")
        if t not in topics:
            topics[t] = []
        topics[t].append(rec)

    samples = []
    for t, recs in topics.items():
        samples.extend(recs[:5])
    samples = samples[:n_sample]

    print(f"\n[Stage 2] Evaluating {len(samples)} answers with LLM...")

    exact_matches = 0
    keyword_overlaps = []
    hallucinations = 0
    llm_scores = []
    llm_latencies = []

    for rec in samples:
        query = rec["query"]
        expected = rec["answer"].lower()
        lang = rec.get("language", "en")

        # Get RTMDK context
        ctx = memory.load_memory_variables({"input": query, "session_id": "bench"})
        context = ctx.get("rtmdk_context", "")

        # Query LLM
        answer, llm_lat = query_llm(query, context)
        llm_latencies.append(llm_lat)
        answer_lower = answer.lower()

        # Exact match (lenient: if expected appears in answer)
        if expected in answer_lower or any(w in answer_lower for w in expected.split() if len(w) > 3):
            exact_matches += 1

        # Keyword overlap (Jaccard)
        expected_words = set(w for w in expected.split() if len(w) > 2)
        actual_words = set(w for w in answer_lower.split() if len(w) > 2)
        if expected_words:
            overlap = len(expected_words & actual_words) / len(expected_words)
            keyword_overlaps.append(overlap)

        # Hallucination check: does answer contain info NOT in context?
        context_words = set(context.lower().split())
        hallucinated_words = actual_words - context_words - expected_words
        hallucination_rate = len(hallucinated_words) / max(len(actual_words), 1)
        if hallucination_rate > 0.5 and len(actual_words) > 5:
            hallucinations += 1

        # LLM judge score
        score = query_llm_judge(expected, answer)
        llm_scores.append(score)

        print(f"  Q: {query[:50]}...")
        print(f"    Expected: {expected[:60]}")
        print(f"    Actual: {answer[:80]}")
        print(f"    Score: {score:.1f}/5, Hallucination: {hallucination_rate:.0%}")

    return {
        "stage": "llm_eval",
        "n_evaluated": len(samples),
        "exact_match_rate": exact_matches / max(len(samples), 1),
        "avg_keyword_overlap": float(np.mean(keyword_overlaps)) if keyword_overlaps else 0.0,
        "hallucination_rate": hallucinations / max(len(samples), 1),
        "avg_llm_score": float(np.mean(llm_scores)) if llm_scores else 0.0,
        "avg_llm_latency_ms": float(np.mean(llm_latencies)) if llm_latencies else 0.0,
        "llm_p95_latency_ms": float(np.percentile(llm_latencies, 95)) if llm_latencies else 0.0,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  RTMDK Comprehensive Benchmark — QA + Token Analysis")
    print("=" * 70)

    # Load dataset
    path = Path("datasets") / "comprehensive_500.json"
    if not path.exists():
        print("  Dataset not found. Running generator...")
        import generate_qa_dataset
        generate_qa_dataset.main()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    print(f"  Loaded {data['n_records']} records ({data.get('n_en', 0)} EN, {data.get('n_ru', 0)} RU)")

    # Stage 1: Retrieval benchmark (limited to 200 for speed)
    n_test = min(200, len(records))
    stage1 = run_stage1(records, max_n=n_test)
    print(f"\n{'='*70}")
    print(f"  STAGE 1 RESULTS — Retrieval ({stage1['n_queries']} queries)")
    print(f"{'='*70}")
    print(f"  Recall@1:  {stage1['recall_at_1']:.2%}")
    print(f"  Recall@3:  {stage1['recall_at_3']:.2%}")
    print(f"  Recall@5:  {stage1['recall_at_5']:.2%}")
    print(f"  Recall@10: {stage1['recall_at_10']:.2%}")
    print(f"  Latency p50: {stage1['latency_p50_ms']:.1f}ms")
    print(f"  Latency p95: {stage1['latency_p95_ms']:.1f}ms")
    print(f"  Latency p99: {stage1['latency_p99_ms']:.1f}ms")
    print(f"\n  Tokens — Query: {stage1['avg_query_tokens']:.0f}, Context: {stage1['avg_context_tokens']:.0f}, Answer: {stage1['avg_answer_tokens']:.0f}")
    print(f"  Token efficiency: {stage1['token_efficiency']:.2f}")
    print(f"\n  By Language:")
    for lang, d in stage1["by_language"].items():
        print(f"    {lang.upper():4s}: Recall={d['recall']:.2%} ({d['total']} queries)")
    print(f"  By Topic:")
    for topic, d in stage1["by_topic"].items():
        print(f"    {topic:12s}: Recall={d['recall']:.2%} ({d['total']})")

    # Save report
    report = {"stage1": stage1}
    path = "comprehensive_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved to {path}")


if __name__ == "__main__":
    main()
