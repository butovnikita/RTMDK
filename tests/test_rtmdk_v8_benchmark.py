"""
test_rtmdk_v8_benchmark.py — FAST Production Benchmark: RTMDK v8.0 vs RAG.

Pre-computes ALL embeddings via batch LM Studio API, then runs benchmark
WITHOUT any HTTP calls during indexing or querying.

Compares 3 methods:
  1. RTMDK v8.0 (resonance-topological memory)
  2. FAISS-like RAG (cosine similarity)
  3. BM25 RAG (text-based keyword retrieval)

Usage:
    python tests/test_rtmdk_v8_benchmark.py [--scale small|medium|large]
"""

import os
import sys
import json
import time
import argparse
import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


# ============================================================================
# BATCH EMBEDDER
# ============================================================================

class BatchEmbedder:
    """Batch embedder using LM Studio API."""

    def __init__(self, url="http://127.0.0.1:12345",
                 model="text-embedding-nomic-embed-text-v1.5"):
        self.url = url
        self.model = model
        self.cache: Dict[str, np.ndarray] = {}
        self._available = self._check()
        if self._available:
            print(f"  LM Studio embedder: {url}")
        else:
            print(f"  LM Studio NOT available, using fallback")

    def _check(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/v1/models", timeout=3)
            return resp.status_code == 200
        except:
            return False

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        """Batch embed texts."""
        if not self._available:
            return [self._fallback(t) for t in texts]

        # Check cache
        uncached = []
        uncached_idx = []
        results = [None] * len(texts)
        for i, t in enumerate(texts):
            if t in self.cache:
                results[i] = self.cache[t]
            else:
                uncached.append(t)
                uncached_idx.append(i)

        if not uncached:
            return results

        # Batch API (50 at a time)
        all_embs = []
        for start in range(0, len(uncached), 50):
            batch = uncached[start:start+50]
            try:
                resp = requests.post(
                    f"{self.url}/v1/embeddings",
                    json={"model": self.model, "input": batch},
                    timeout=60,
                )
                data = resp.json()
                for item in data.get("data", []):
                    idx = item.get("index", 0)
                    emb = np.array(item["embedding"], dtype=np.float32)
                    all_embs.append(emb)
                    self.cache[batch[idx]] = emb
            except Exception as e:
                print(f"  Batch error: {e}")
                for t in batch:
                    fb = self._fallback(t)
                    all_embs.append(fb)
                    self.cache[t] = fb

        for i, idx in enumerate(uncached_idx):
            if i < len(all_embs):
                results[idx] = all_embs[i]

        return results

    def __call__(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        return self.embed_many([text])[0]

    @staticmethod
    def _fallback(text: str, dim: int = 768) -> np.ndarray:
        np.random.seed(42)
        base = np.random.randn(dim).astype(np.float32) * 0.01
        for tok in text.lower().split()[:20]:
            np.random.seed(hash(tok + "bench_seed") % 2**32)
            d = np.random.randn(dim).astype(np.float32)
            base += (d / (np.linalg.norm(d) + 1e-8)) * 0.5
        return base


# ============================================================================
# DATA
# ============================================================================

def load_and_scale(path: str, target: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    if target <= len(records):
        return records[:target]

    scaled = list(records)
    n_passes = max(1, target // len(records))
    for p in range(1, n_passes):
        for rec in records:
            if len(scaled) >= target:
                break
            scaled.append({
                "query": rec["query"], "answer": rec["answer"],
                "context": f"[v{p+1}] {rec['context']}",
                "topic": rec.get("topic", "general"),
            })

    topics = list(set(r.get("topic", "general") for r in records))
    while len(scaled) < target:
        i = len(scaled) - len(records)
        scaled.append({
            "query": f"Noise query {i}", "answer": f"noise{i}",
            "context": f"Noise document #{i} for topic '{topics[i % len(topics)]}' with generic content.",
            "topic": topics[i % len(topics)],
        })
    return scaled[:target]


# ============================================================================
# RETRIEVERS
# ============================================================================

class FAISSRAG:
    def __init__(self, embeddings: np.ndarray, texts: List[str]):
        self.embeddings = embeddings
        self.texts = texts

    def retrieve(self, query_emb: np.ndarray, top_k: int = 10) -> List[Tuple[str, float, str]]:
        if len(self.embeddings) == 0:
            return []
        q_norm = np.linalg.norm(query_emb) + 1e-8
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        sims = (self.embeddings @ query_emb) / (norms.flatten() * q_norm)
        top = np.argsort(sims)[::-1][:top_k]
        return [(f"doc_{i}", float(sims[i]), self.texts[i]) for i in top]


class BM25RAG:
    def __init__(self, texts: List[str], k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.texts = texts
        self.doc_tokens = [re.findall(r'\b\w{2,}\b', t.lower()) for t in texts]
        self._build_index()

    def _build_index(self):
        n = len(self.doc_tokens)
        if n == 0:
            return
        self.avg_dl = np.mean([len(d) for d in self.doc_tokens])
        df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        qt = re.findall(r'\b\w{2,}\b', query.lower())
        if not qt or not self.doc_tokens:
            return []
        scores = []
        for i, dt in enumerate(self.doc_tokens):
            s = 0.0
            dl = len(dt)
            for t in qt:
                if t in self.idf:
                    tf = dt.count(t)
                    num = tf * (self.k1 + 1)
                    den = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                    s += self.idf[t] * num / den
            scores.append((f"doc_{i}", s, self.texts[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================================
# METRICS
# ============================================================================

def compute_recall(results, answer: str) -> Dict[str, bool]:
    words = [w.lower() for w in re.findall(r'\b\w{2,}\b', answer) if len(w) > 2]
    if not words:
        return {"r1": True, "r3": True, "r5": True, "r10": True}
    found = {}
    for rank, (_, _, text) in enumerate(results[:10], 1):
        if any(w in text.lower() for w in words):
            found[rank] = True
    return {
        "r1": 1 in found, "r3": any(r <= 3 for r in found),
        "r5": any(r <= 5 for r in found), "r10": any(r <= 10 for r in found),
    }


# ============================================================================
# MAIN
# ============================================================================

def run_benchmark(scale: str = "small", dataset_path: str = "datasets/qa_1000_en.json",
                  results_dir: str = "tests/results"):
    scale_map = {"small": 1000, "medium": 5000, "large": 10000}
    target = scale_map.get(scale, 1000)

    print("=" * 70)
    print(f"  RTMDK v8.0 vs RAG — {scale.upper()} ({target} nodes)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load data
    print(f"\n[1] Loading & scaling dataset...")
    records = load_and_scale(dataset_path, target)
    print(f"  {len(records)} records")

    # Pre-compute embeddings (BATCH)
    print(f"\n[2] Pre-computing embeddings (batch LM Studio API)...")
    embedder = BatchEmbedder()

    all_texts = [r["context"] for r in records]
    all_queries = [r["query"] for r in records]

    t0_emb = time.perf_counter()
    print(f"  Embedding {len(all_texts)} contexts...")
    ctx_embs = embedder.embed_many(all_texts)
    print(f"  Embedding {len(all_queries)} queries...")
    q_embs = embedder.embed_many(all_queries)
    emb_time = time.perf_counter() - t0_emb
    print(f"  Embedding complete: {emb_time:.1f}s ({len(ctx_embs)+len(q_embs)} vectors)")

    # Initialize methods
    print(f"\n[3] Initializing retrieval methods...")

    # RTMDK — use pre-computed embeddings
    memory = RTMDKMemory(
        config=RTMDKConfig(
            embedding_dim=768, latent_dim=64, top_k=15, min_response=0.005,
            decay_rate=0.999, use_hnsw=(target > 500), learn_projection=False,
            bm25_fallback=False, enable_async=False, attention_bias=True,
            context_format="attention",
        ),
        embedder=embedder,
    )

    # Pre-populate RTMDK with pre-computed embeddings
    print(f"  Populating RTMDK nodes directly...")
    for i, (ctx, emb) in enumerate(zip(all_texts, ctx_embs)):
        nid = f"doc_{i}"
        memory.field.add_node(emb, {"text": ctx, "topic": records[i].get("topic", "")})
    print(f"  RTMDK: {len(memory.field.nodes)} nodes")

    # FAISS
    faiss = FAISSRAG(np.array(ctx_embs), all_texts)
    print(f"  FAISS: {faiss.embeddings.shape[0]} docs")

    # BM25
    bm25 = BM25RAG(all_texts)
    print(f"  BM25: {len(bm25.texts)} docs")

    # RTMDK consolidation
    print(f"\n[4] RTMDK consolidation...")
    t0 = time.perf_counter()
    for _ in range(3):
        memory.field.step()
    cons_time = time.perf_counter() - t0
    print(f"  Consolidation: {cons_time:.1f}s")

    # Run queries (NO HTTP calls — embeddings pre-computed)
    print(f"\n[5] Running {len(records)} queries...")
    methods = {"RTMDK v8.0": memory, "FAISS RAG": faiss, "BM25 RAG": bm25}
    all_results = {name: {"r1": 0, "r3": 0, "r5": 0, "r10": 0, "latencies": [], "exact": 0} for name in methods}
    t0_q = time.perf_counter()

    for qi, rec in enumerate(records):
        query = rec["query"]
        answer = rec["answer"]
        q_emb = q_embs[qi]

        # RTMDK — use pre-computed embedding (NO HTTP call)
        t0 = time.perf_counter()
        ctx = memory.load_memory_variables_with_embedding(
            {"input": query, "session_id": "default"}, q_emb
        )
        rtmdk_context = ctx.get("rtmdk_context", "")
        rtmdk_results = []
        for line in rtmdk_context.split("\n"):
            if line.startswith("[ATTN:"):
                m = re.match(r'\[ATTN:([\d.]+)\]\[SAL:([\d.]+)\]\[TIER:(\w)\]\s*(.*)', line)
                if m:
                    rtmdk_results.append(("attn", float(m.group(1))*float(m.group(2)), m.group(4)))
        latency = (time.perf_counter() - t0) * 1000
        recall = compute_recall(rtmdk_results, answer)
        for k in ["r1","r3","r5","r10"]:
            all_results["RTMDK v8.0"][k] += recall[k]
        all_results["RTMDK v8.0"]["latencies"].append(latency)
        if rtmdk_results and any(w.lower() in " ".join(t for _,_,t in rtmdk_results).lower() for w in answer.split() if len(w)>2):
            all_results["RTMDK v8.0"]["exact"] += 1

        # FAISS
        t0 = time.perf_counter()
        faiss_results = faiss.retrieve(q_emb, top_k=10)
        latency = (time.perf_counter() - t0) * 1000
        recall = compute_recall(faiss_results, answer)
        for k in ["r1","r3","r5","r10"]:
            all_results["FAISS RAG"][k] += recall[k]
        all_results["FAISS RAG"]["latencies"].append(latency)
        if faiss_results and any(w.lower() in " ".join(t for _,_,t in faiss_results).lower() for w in answer.split() if len(w)>2):
            all_results["FAISS RAG"]["exact"] += 1

        # BM25
        t0 = time.perf_counter()
        bm25_results = bm25.retrieve(query, top_k=10)
        latency = (time.perf_counter() - t0) * 1000
        recall = compute_recall(bm25_results, answer)
        for k in ["r1","r3","r5","r10"]:
            all_results["BM25 RAG"][k] += recall[k]
        all_results["BM25 RAG"]["latencies"].append(latency)
        if bm25_results and any(w.lower() in " ".join(t for _,_,t in bm25_results).lower() for w in answer.split() if len(w)>2):
            all_results["BM25 RAG"]["exact"] += 1

        if (qi + 1) % 200 == 0:
            n = qi + 1
            print(f"  Query {n}/{len(records)} ({time.perf_counter()-t0_q:.0f}s):")
            for name in methods:
                r = all_results[name]
                print(f"    {name:15s}: R@1={r['r1']/n:.0%}  R@5={r['r5']/n:.0%}  P95={np.percentile(r['latencies'], 95):.0f}ms")

    # ── Report ──
    n = len(records)
    print(f"\n{'='*70}")
    print(f"  RESULTS — RTMDK v8.0 vs RAG ({scale}, {n} nodes)")
    print(f"{'='*70}")
    print(f"\n  {'Method':<15} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'P50':>6} {'P95':>6} {'Exact':>6}")
    print(f"  {'='*65}")
    for name in methods:
        r = all_results[name]
        print(f"  {name:<15} {r['r1']/n:>5.0%} {r['r3']/n:>5.0%} {r['r5']/n:>5.0%} {r['r10']/n:>5.0%} "
              f"{np.percentile(r['latencies'], 50):>4.0f}ms {np.percentile(r['latencies'], 95):>4.0f}ms "
              f"{r['exact']/n:>5.0%}")

    best = max(methods, key=lambda n: all_results[n]["r1"])
    print(f"\n  🏆 Best R@1: {best} ({all_results[best]['r1']/n:.0%})")

    # Save
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f"v8_benchmark_{scale}_{ts}.json")
    report = {
        "scale": scale, "target_nodes": target, "n_queries": n,
        "embedding_time_s": round(emb_time, 1),
        "consolidation_time_s": round(cons_time, 1),
        "embedder": "LM Studio (real)" if embedder._available else "fallback",
        "timestamp": ts,
        "results": {
            name: {
                "recall_at_1": round(r["r1"]/n, 4), "recall_at_3": round(r["r3"]/n, 4),
                "recall_at_5": round(r["r5"]/n, 4), "recall_at_10": round(r["r10"]/n, 4),
                "latency_p50_ms": round(float(np.percentile(r["latencies"], 50)), 2),
                "latency_p95_ms": round(float(np.percentile(r["latencies"], 95)), 2),
                "latency_p99_ms": round(float(np.percentile(r["latencies"], 99)), 2),
                "latency_mean_ms": round(float(np.mean(r["latencies"])), 2),
                "exact_match_rate": round(r["exact"]/n, 4),
            } for name, r in all_results.items()
        },
    }

    # Compare with previous
    prev_path = "archive/reports/final_benchmark_report.json"
    if os.path.exists(prev_path):
        with open(prev_path) as f:
            prev = json.load(f)
        print(f"\n  {'='*70}")
        print(f"  vs Previous (1000 nodes):")
        print(f"  {'='*70}")
        for pn in prev.get("results", {}):
            cn = pn.replace("RTMDK ", "").replace("Optimized", "v8.0")
            if cn in report["results"]:
                pr = prev["results"][pn]
                cr = report["results"][cn]
                d = ((cr["recall_at_1"] - pr["recall_at_1"]) / max(pr["recall_at_1"], 0.01)) * 100
                print(f"  {pn}: R@1 {pr['recall_at_1']:.1%} → {cr['recall_at_1']:.1%} ({d:+.1f}%)")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--dataset", default="datasets/qa_1000_en.json")
    args = parser.parse_args()
    run_benchmark(scale=args.scale, dataset_path=args.dataset)
