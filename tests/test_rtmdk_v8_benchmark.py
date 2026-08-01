"""
test_rtmdk_v8_benchmark.py — Production Benchmark: RTMDK v8.0 vs RAG.

Uses REAL LM Studio embedder and REAL dataset (qa_1000_en.json).
Compares 3 retrieval methods:
  1. RTMDK v8.0 (resonance-topological memory)
  2. FAISS-like RAG (cosine similarity in embedding space)
  3. BM25 RAG (text-based keyword retrieval)

Scales: small (1K nodes), medium (5K nodes), large (10K nodes)

Usage:
    python tests/test_rtmdk_v8_benchmark.py [--scale small|medium|large]
"""

from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
import os
import sys
import json
import time
import argparse
import re
import math
from collections import Counter
from datetime import datetime
import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BatchEmbedder:
    def __init__(self, url="http://127.0.0.1:12345", model="text-embedding-nomic-embed-text-v1.5"):
        self.url = url
        self.model = model
        self.cache = {}
        self._available = self._check()
        print("  Embedder:", "LM Studio (real)" if self._available else "fallback")

    def _check(self):
        try:
            return requests.get(f"{self.url}/v1/models", timeout=3).status_code == 200
        except BaseException:
            return False

    def embed_many(self, texts):
        if not self._available:
            return [self._fallback(t) for t in texts]
        uncached, uncached_idx, results = [], [], [None] * len(texts)
        for i, t in enumerate(texts):
            if t in self.cache:
                results[i] = self.cache[t]
            else:
                uncached.append(t)
                uncached_idx.append(i)
        if not uncached:
            return results
        all_embs = []
        for start in range(0, len(uncached), 50):
            batch = uncached[start : start + 50]
            try:
                resp = requests.post(
                    f"{self.url}/v1/embeddings", json={"model": self.model, "input": batch}, timeout=60
                )
                for item in resp.json().get("data", []):
                    idx = item.get("index", 0)
                    emb = np.array(item["embedding"], dtype=np.float32)
                    all_embs.append(emb)
                    self.cache[batch[idx]] = emb
            except Exception as e:
                print("  Batch error:", e)
                for t in batch:
                    fb = self._fallback(t)
                    all_embs.append(fb)
                    self.cache[t] = fb
        for i, idx in enumerate(uncached_idx):
            if i < len(all_embs):
                results[idx] = all_embs[i]
        return results

    def __call__(self, text):
        if text in self.cache:
            return self.cache[text]
        return self.embed_many([text])[0]

    @staticmethod
    def _fallback(text, dim=768):
        rng = np.random.default_rng(42)
        base = rng.standard_normal(dim).astype(np.float32) * 0.01
        for tok in text.lower().split()[:20]:
            tok_rng = np.random.default_rng(hash(tok + "bench_seed") % 2**32)
            d = tok_rng.standard_normal(dim).astype(np.float32)
            base += (d / (np.linalg.norm(d) + 1e-8)) * 0.5
        return base


def load_and_scale(path, target):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    if target <= len(records):
        return records[:target]
    scaled = list(records)
    for p in range(1, max(1, target // len(records))):
        for rec in records:
            if len(scaled) >= target:
                break
            scaled.append(
                {
                    "query": rec["query"],
                    "answer": rec["answer"],
                    "context": "[v%d] %s" % (p + 1, rec["context"]),
                    "topic": rec.get("topic", "general"),
                }
            )
    topics = list(set(r.get("topic", "general") for r in records))
    while len(scaled) < target:
        i = len(scaled) - len(records)
        scaled.append(
            {
                "query": "Noise %d" % i,
                "answer": "noise%d" % i,
                "context": "Noise doc #%d for topic '%s'." % (i, topics[i % len(topics)]),
                "topic": topics[i % len(topics)],
            }
        )
    return scaled[:target]


class FAISSRAG:
    def __init__(self, embeddings, texts):
        self.embeddings = embeddings
        self.texts = texts

    def retrieve(self, query_emb, top_k=10):
        if len(self.embeddings) == 0:
            return []
        q_norm = np.linalg.norm(query_emb) + 1e-8
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        sims = (self.embeddings @ query_emb) / (norms.flatten() * q_norm)
        top = np.argsort(sims)[::-1][:top_k]
        return [("doc_%d" % i, float(sims[i]), self.texts[i]) for i in top]


class BM25RAG:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b, self.texts = k1, b, texts
        self.doc_tokens = [re.findall(r"\b\w{2,}\b", t.lower()) for t in texts]
        n = len(self.doc_tokens)
        self.avg_dl = np.mean([len(d) for d in self.doc_tokens]) if n else 1
        df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def retrieve(self, query, top_k=10):
        qt = re.findall(r"\b\w{2,}\b", query.lower())
        if not qt or not self.doc_tokens:
            return []
        scores = []
        for i, dt in enumerate(self.doc_tokens):
            s = 0.0
            dl = len(dt)
            for t in qt:
                if t in self.idf:
                    tf = dt.count(t)
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
            scores.append(("doc_%d" % i, s, self.texts[i]))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def compute_recall(results, answer):
    words = [w.lower() for w in re.findall(r"\b\w{2,}\b", answer) if len(w) > 2]
    if not words:
        return {"r1": True, "r3": True, "r5": True, "r10": True}
    found = {}
    for rank, (_, _, text) in enumerate(results[:10], 1):
        if any(w in text.lower() for w in words):
            found[rank] = True
    return {
        "r1": 1 in found,
        "r3": any(r <= 3 for r in found),
        "r5": any(r <= 5 for r in found),
        "r10": any(r <= 10 for r in found),
    }


def run_benchmark(scale="small", dataset_path="datasets/qa_1000_en.json", results_dir="tests/results"):
    scale_map = {"small": 1000, "medium": 5000, "large": 10000}
    target = scale_map.get(scale, 1000)
    print("=" * 70)
    print("  RTMDK v8.0 vs RAG - %s (%d nodes)" % (scale.upper(), target))
    print("  %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("=" * 70)

    print("\n[1] Loading dataset...")
    records = load_and_scale(dataset_path, target)
    print("  %d records" % len(records))

    print("\n[2] Pre-computing embeddings...")
    embedder = BatchEmbedder()
    t0_emb = time.perf_counter()
    ctx_embs = embedder.embed_many([r["context"] for r in records])
    q_embs = embedder.embed_many([r["query"] for r in records])
    emb_time = time.perf_counter() - t0_emb
    print("  Embedding done: %.1fs (%d vectors)" % (emb_time, len(ctx_embs) + len(q_embs)))

    print("\n[3] Initializing retrieval...")
    # Production preset benchmark
    # Use production preset with benchmark-specific overrides
    cfg = RTMDKConfig.production()
    cfg.top_k = 15
    cfg.use_hnsw = target > 5000
    cfg.enable_async = False
    cfg.phase_coupling = 0.0  # Disable phase noise for pure retrieval benchmarks
    cfg.consolidation_async = True
    memory = RTMDKMemory(config=cfg, embedder=embedder)
    print("  Populating RTMDK...")
    for i, (ctx, emb) in enumerate(zip([r["context"] for r in records], ctx_embs)):
        memory.field.add_node(emb, {"text": ctx, "topic": records[i].get("topic", "")})
    print("  RTMDK: %d nodes" % len(memory.field.nodes))
    faiss = FAISSRAG(np.array(ctx_embs), [r["context"] for r in records])
    bm25 = BM25RAG([r["context"] for r in records])

    print("\n[4] RTMDK consolidation...")
    t0 = time.perf_counter()
    for _ in range(3):
        memory.field.step()
    cons_time = time.perf_counter() - t0
    print("  Consolidation: %.1fs" % cons_time)

    print("\n[5] Running %d queries..." % len(records))
    methods = {"RTMDK v8.0": memory, "FAISS RAG": faiss, "BM25 RAG": bm25}
    all_results = {name: {"r1": 0, "r3": 0, "r5": 0, "r10": 0, "latencies": [], "exact": 0} for name in methods}
    t0_q = time.perf_counter()

    for qi, rec in enumerate(records):
        query, answer, q_emb = rec["query"], rec["answer"], q_embs[qi]

        # RTMDK
        t0 = time.perf_counter()
        ctx = memory.load_memory_variables_with_embedding({"input": query, "session_id": "default"}, q_emb)
        latency = (time.perf_counter() - t0) * 1000
        rtmdk_results = []
        for line in ctx.get("rtmdk_context", "").split("\n"):
            if line.startswith("[ATTN:"):
                m = re.match(r"\[ATTN:([\d.]+)\]\[SAL:([\d.]+)\]\[TIER:(\w)\]\s*(.*)", line)
                if m:
                    rtmdk_results.append(("attn", float(m.group(1)) * float(m.group(2)), m.group(4)))
        recall = compute_recall(rtmdk_results, answer)
        for k in ["r1", "r3", "r5", "r10"]:
            all_results["RTMDK v8.0"][k] += recall[k]
        all_results["RTMDK v8.0"]["latencies"].append(latency)
        if rtmdk_results and any(
            w.lower() in " ".join(t for _, _, t in rtmdk_results).lower() for w in answer.split() if len(w) > 2
        ):
            all_results["RTMDK v8.0"]["exact"] += 1

        # FAISS
        t0 = time.perf_counter()
        faiss_results = faiss.retrieve(q_emb, top_k=10)
        latency = (time.perf_counter() - t0) * 1000
        recall = compute_recall(faiss_results, answer)
        for k in ["r1", "r3", "r5", "r10"]:
            all_results["FAISS RAG"][k] += recall[k]
        all_results["FAISS RAG"]["latencies"].append(latency)
        if faiss_results and any(
            w.lower() in " ".join(t for _, _, t in faiss_results).lower() for w in answer.split() if len(w) > 2
        ):
            all_results["FAISS RAG"]["exact"] += 1

        # BM25
        t0 = time.perf_counter()
        bm25_results = bm25.retrieve(query, top_k=10)
        latency = (time.perf_counter() - t0) * 1000
        recall = compute_recall(bm25_results, answer)
        for k in ["r1", "r3", "r5", "r10"]:
            all_results["BM25 RAG"][k] += recall[k]
        all_results["BM25 RAG"]["latencies"].append(latency)
        if bm25_results and any(
            w.lower() in " ".join(t for _, _, t in bm25_results).lower() for w in answer.split() if len(w) > 2
        ):
            all_results["BM25 RAG"]["exact"] += 1

        if (qi + 1) % 200 == 0:
            n = qi + 1
            print("  Query %d/%d (%.0fs):" % (n, len(records), time.perf_counter() - t0_q))
            for name in methods:
                r = all_results[name]
                print(
                    "    %s: R@1=%.0f%% R@5=%.0f%% P95=%.0fms"
                    % (name, r["r1"] / n * 100, r["r5"] / n * 100, np.percentile(r["latencies"], 95))
                )

    # Save report
    n = len(records)
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, "v8_benchmark_%s_%s.json" % (scale, ts))
    report = {
        "scale": scale,
        "target_nodes": target,
        "n_queries": n,
        "embedding_time_s": round(emb_time, 1),
        "consolidation_time_s": round(cons_time, 1),
        "embedder": "LM Studio (real)" if embedder._available else "fallback",
        "timestamp": ts,
        "results": {
            name: {
                "recall_at_1": round(r["r1"] / n, 4),
                "recall_at_3": round(r["r3"] / n, 4),
                "recall_at_5": round(r["r5"] / n, 4),
                "recall_at_10": round(r["r10"] / n, 4),
                "latency_p50_ms": round(float(np.percentile(r["latencies"], 50)), 2),
                "latency_p95_ms": round(float(np.percentile(r["latencies"], 95)), 2),
                "latency_p99_ms": round(float(np.percentile(r["latencies"], 99)), 2),
                "latency_mean_ms": round(float(np.mean(r["latencies"])), 2),
                "exact_match_rate": round(r["exact"] / n, 4),
            }
            for name, r in all_results.items()
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nRESULTS: %s %d nodes" % (scale, n))
    for name in methods:
        r = report["results"][name]
        print(
            "  %s: R@1=%.1f%% R@5=%.1f%% P50=%.0fms P95=%.0fms Exact=%.1f%%"
            % (
                name,
                r["recall_at_1"] * 100,
                r["recall_at_5"] * 100,
                r["latency_p50_ms"],
                r["latency_p95_ms"],
                r["exact_match_rate"] * 100,
            )
        )
    best = max(methods, key=lambda n: all_results[n]["r1"])
    print("  Best R@1: %s %.1f%%" % (best, all_results[best]["r1"] / n * 100))
    print("  Report:", report_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--dataset", default="datasets/qa_1000_en.json")
    args = parser.parse_args()
    run_benchmark(scale=args.scale, dataset_path=args.dataset)
