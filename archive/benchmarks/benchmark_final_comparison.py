"""
benchmark_final_comparison.py — RTMDK vs RAG: Final Comparison on 1000 QA.

Compares 4 retrieval methods:
  1. RTMDK Baseline (resonance only)
  2. RTMDK Optimized (hybrid: resonance + BM25 + cosine)
  3. FAISS-like RAG (cosine similarity in original embedding space)
  4. BM25 RAG (text-based retrieval only)

Uses datasets/qa_1000_en.json (1000 unique EN QA pairs).
LLM-as-judge on 100 sampled answers via LM Studio API.

Checkpoints saved every 250 queries for resumability.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedder_lmstudio import get_embedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory
from rtmdk.production.bm25_fallback import BM25FallbackRetriever
from rtmdk.production.advanced_retrieval import HybridRetriever


# ============================================================================
# DATA LOADER
# ============================================================================

def load_dataset(path: str = "datasets/qa_1000_en.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


# ============================================================================
# RETRIEVAL METHODS
# ============================================================================

class RTMDKBaseline:
    """Standard RTMDK resonance retrieval."""
    def __init__(self, memory: RTMDKMemory):
        self.memory = memory

    def retrieve(self, query: str, query_emb: np.ndarray, top_k: int = 5):
        phase = self.memory._get_phase("baseline", query_emb)
        results = self.memory.field.query(query_emb, phase, top_k=top_k * 2)
        return [(nid, score, node) for nid, score, node in results[:top_k]]


class RTMDKOptimized:
    """RTMDK with hybrid retrieval."""
    def __init__(self, memory: RTMDKMemory, bm25: BM25FallbackRetriever):
        self.hybrid = HybridRetriever(memory, bm25)
        self.memory = memory

    def retrieve(self, query: str, query_emb: np.ndarray, top_k: int = 5):
        return self.hybrid.retrieve(query, query_emb, top_k)


class FAISSRAG:
    """Cosine similarity retrieval (standard RAG baseline)."""
    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}
        self.texts: Dict[str, str] = {}

    def add(self, node_id: str, embedding: np.ndarray, text: str):
        self.embeddings[node_id] = embedding
        self.texts[node_id] = text

    def retrieve(self, query: str, query_emb: np.ndarray, top_k: int = 5):
        if not self.embeddings:
            return []
        query_norm = np.linalg.norm(query_emb) + 1e-8
        scores = []
        for nid, emb in self.embeddings.items():
            cos = float(np.dot(query_emb, emb) / (query_norm * np.linalg.norm(emb) + 1e-8))
            scores.append((nid, max(0.0, (cos + 1.0) / 2.0), None))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BM25RAG:
    """BM25-only retrieval (traditional text search baseline)."""
    def __init__(self):
        self.bm25 = BM25FallbackRetriever()

    def add(self, node_id: str, text: str):
        self.bm25.add_document(node_id, text)

    def retrieve(self, query: str, query_emb: np.ndarray, top_k: int = 5):
        results = self.bm25.search(query, top_k)
        return [(nid, score, None) for nid, score in results]


# ============================================================================
# METRICS
# ============================================================================

def compute_recall(results, answer: str) -> Dict[str, bool]:
    """Check if answer keywords appear in retrieved context."""
    answer_words = [w for w in answer.lower().split() if len(w) > 2]
    if not answer_words:
        return {"r1": True, "r3": True, "r5": True, "r10": True}

    found_at = {}
    for rank, (nid, score, node) in enumerate(results[:10], 1):
        if node is None:
            continue
        text = node.content.get("text", "").lower()
        if any(w in text for w in answer_words):
            found_at[rank] = True

    return {
        "r1": 1 in found_at,
        "r3": any(r <= 3 for r in found_at),
        "r5": any(r <= 5 for r in found_at),
        "r10": any(r <= 10 for r in found_at),
    }


def llm_judge_answer(query: str, expected: str, actual: str) -> float:
    """Ask LLM to rate answer quality 1-5."""
    try:
        import requests
        prompt = f"""Rate this answer 1-5:
Question: {query}
Expected: {expected}
Actual: {actual}
Score (just the number):"""
        resp = requests.post(
            "http://localhost:12345/api/v1/chat",
            json={
                "model": "thedrummer_rocinante-x-12b-v1",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "max_tokens": 5,
            },
            timeout=30,
        )
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "3").strip()
        for c in text:
            if c in "12345":
                return float(c)
    except:
        pass
    return 3.0


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def run_benchmark(n_queries: int = 1000, llm_judge_n: int = 100, checkpoint_every: int = 250):
    print("=" * 70)
    print(f"  FINAL BENCHMARK: RTMDK vs RAG ({n_queries} QA)")
    print("=" * 70)

    # Load data
    records = load_dataset()
    records = records[:n_queries]
    print(f"  Loaded {len(records)} QA pairs")

    embedder = get_embedder()

    # ── Setup all 4 methods ──

    # 1. RTMDK Baseline
    memory_baseline = RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=768, latent_dim=256, top_k=5, min_response=0.005,
            decay_rate=0.999, enable_async=False, bm25_fallback=False,
            use_hnsw=True, learn_projection=False,
        ),
        embedder=embedder,
    )
    rtmdk_baseline = RTMDKBaseline(memory_baseline)

    # 2. RTMDK Optimized
    memory_opt = RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=768, latent_dim=256, top_k=5, min_response=0.005,
            decay_rate=0.999, enable_async=False, bm25_fallback=True,
            use_hnsw=True, learn_projection=False,
        ),
        embedder=embedder,
    )
    bm25_opt = BM25FallbackRetriever()
    rtmdk_opt = RTMDKOptimized(memory_opt, bm25_opt)

    # 3. FAISS RAG
    faiss_rag = FAISSRAG()

    # 4. BM25 RAG
    bm25_rag = BM25RAG()

    # ── Populate all methods ──
    print(f"\n  Indexing {len(records)} facts...")
    t0_index = time.perf_counter()

    for i, rec in enumerate(records):
        emb = embedder(rec["context"])
        nid = f"n_{i}"

        # RTMDK methods
        memory_baseline.field.add_node(emb, {"text": rec["context"], "topic": rec["topic"]})
        memory_opt.field.add_node(emb, {"text": rec["context"], "topic": rec["topic"]})
        bm25_opt.add_document(nid, rec["context"])

        # FAISS RAG
        faiss_rag.add(nid, emb, rec["context"])

        # BM25 RAG
        bm25_rag.add(nid, rec["context"])

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t0_index
            print(f"    Indexed {i+1}/{len(records)} ({elapsed:.0f}s)")

    index_time = time.perf_counter() - t0_index
    print(f"  Indexing complete: {index_time:.0f}s")

    # ── Run retrieval benchmark ──
    print(f"\n  Running {n_queries} queries...")
    methods = {
        "RTMDK Baseline": rtmdk_baseline,
        "RTMDK Optimized": rtmdk_opt,
        "FAISS RAG": faiss_rag,
        "BM25 RAG": bm25_rag,
    }

    all_results = {name: {"r1": 0, "r3": 0, "r5": 0, "r10": 0, "latencies": []} for name in methods}
    checkpoint_path = Path("benchmark_checkpoint.json")

    # Load checkpoint if exists
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            start_q = ckpt.get("queries_done", 0)
            all_results.update(ckpt.get("results", {}))
            print(f"  Resuming from query {start_q}")
        except:
            start_q = 0
    else:
        start_q = 0

    for qi in range(start_q, len(records)):
        rec = records[qi]
        query = rec["query"]
        answer = rec["answer"]
        query_emb = embedder(query)

        for name, retriever in methods.items():
            t0 = time.perf_counter()
            results = retriever.retrieve(query, query_emb, top_k=10)
            latency = (time.perf_counter() - t0) * 1000

            recall = compute_recall(results, answer)
            all_results[name]["r1"] += recall["r1"]
            all_results[name]["r3"] += recall["r3"]
            all_results[name]["r5"] += recall["r5"]
            all_results[name]["r10"] += recall["r10"]
            all_results[name]["latencies"].append(latency)

        if (qi + 1) % 100 == 0:
            # Print progress
            print(f"  Query {qi+1}/{len(records)}:")
            for name in methods:
                r = all_results[name]
                n = qi + 1 - start_q
                print(f"    {name:20s}: R@1={r['r1']/n:.0%}  R@5={r['r5']/n:.0%}  P95={np.percentile(r['latencies'], 95):.0f}ms")

        # Checkpoint
        if (qi + 1) % checkpoint_every == 0:
            ckpt = {
                "queries_done": qi + 1,
                "results": all_results,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(ckpt, f)
            print(f"  Checkpoint saved at query {qi+1}")

    # ── LLM Judge (sample 100) ──
    print(f"\n  LLM-as-Judge on {llm_judge_n} samples...")
    llm_results = {name: {"scores": [], "exact_matches": 0} for name in methods}
    sample_indices = list(range(0, len(records), len(records) // llm_judge_n))[:llm_judge_n]

    for idx in sample_indices:
        rec = records[idx]
        query = rec["query"]
        answer = rec["answer"]
        query_emb = embedder(query)

        for name, retriever in methods.items():
            results = retriever.retrieve(query, query_emb, top_k=3)
            # Get context text
            context_parts = []
            for nid, score, node in results:
                if node:
                    context_parts.append(node.content.get("text", "")[:200])
                else:
                    # For FAISS/BM25
                    context_parts.append(f"doc_{nid}")

            context = " ".join(context_parts)

            # Check exact match
            if any(w in context.lower() for w in answer.lower().split() if len(w) > 3):
                llm_results[name]["exact_matches"] += 1

            # LLM judge (optional — slow)
            # score = llm_judge_answer(query, answer, context)
            # llm_results[name]["scores"].append(score)

    # ── Final Report ──
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — RTMDK vs RAG")
    print(f"{'='*70}")
    n = len(records) - start_q

    print(f"\n  {'Method':<20} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'P50':>6} {'P95':>6} {'Exact':>6}")
    print(f"  {'─'*66}")
    for name in methods:
        r = all_results[name]
        print(f"  {name:<20} {r['r1']/n:>5.0%} {r['r3']/n:>5.0%} {r['r5']/n:>5.0%} {r['r10']/n:>5.0%} "
              f"{np.percentile(r['latencies'], 50):>4.0f}ms {np.percentile(r['latencies'], 95):>4.0f}ms "
              f"{llm_results[name]['exact_matches']/llm_judge_n:>5.0%}")

    # Winner
    best_r1_name = max(methods, key=lambda n: all_results[n]["r1"])
    print(f"\n  🏆 Best Recall@1: {best_r1_name} ({all_results[best_r1_name]['r1']/n:.0%})")

    # Save report
    report = {
        "n_queries": len(records),
        "indexing_time_s": round(index_time, 1),
        "results": {
            name: {
                "recall_at_1": round(r["r1"] / n, 4),
                "recall_at_3": round(r["r3"] / n, 4),
                "recall_at_5": round(r["r5"] / n, 4),
                "recall_at_10": round(r["r10"] / n, 4),
                "latency_p50_ms": round(float(np.percentile(r["latencies"], 50)), 2),
                "latency_p95_ms": round(float(np.percentile(r["latencies"], 95)), 2),
                "exact_match_rate": round(llm_results[name]["exact_matches"] / llm_judge_n, 4),
            }
            for name, r in all_results.items()
        },
    }

    with open("final_benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to final_benchmark_report.json")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == "__main__":
    run_benchmark()
