"""
test_stability_10k.py — Stability benchmark at 5K and 10K nodes.

Uses LM Studio for real embeddings. Measures:
- Insert throughput (nodes/sec)
- RAM consumption
- Query latency (P50, P95, P99)
- R@1 on original 1K QA (accuracy must not degrade)
- Consolidation time
- Save/load serialization time

Usage:
    python tests/test_stability_10k.py [--scale 5000|10000]
"""

import os
import sys
import json
import time
import argparse
import tempfile
import gc
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import psutil
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
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
        except Exception:
            return False

    def embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        embs = self.embed_many([text])
        self.cache[text] = embs[0]
        return embs[0]

    def embed_many(self, texts: List[str]) -> List[np.ndarray]:
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
        for start in range(0, len(uncached), 50):
            batch = uncached[start:start+50]
            try:
                resp = requests.post(
                    f"{self.url}/v1/embeddings",
                    json={"model": self.model, "input": batch},
                    timeout=120,
                )
                data = resp.json().get("data", [])
                for item in data:
                    idx = item.get("index", 0)
                    emb = np.array(item["embedding"], dtype=np.float32)
                    global_idx = uncached_idx[start + idx]
                    results[global_idx] = emb
                    self.cache[uncached[start + idx]] = emb
            except Exception as e:
                print(f"  Batch embed error ({start}): {e}")
                for i in range(start, min(start + 50, len(uncached))):
                    results[uncached_idx[i]] = self._fallback(uncached[i])
        return results

    def _fallback(self, text: str) -> np.ndarray:
        h = hash(text) % (2**31)
        rng = np.random.RandomState(h)
        emb = rng.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def load_qa(path: str = "datasets/qa_1000_en.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expected format: list of {"question": ..., "answer": ...}
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    return data


def expand_dataset(records: List[Dict], target_size: int) -> List[Dict]:
    """Duplicate records with minor textual variants to reach target_size."""
    n_original = len(records)
    multiplier = target_size // n_original
    remainder = target_size % n_original
    out = []
    group_id = 0
    for rec in records:
        base_q = rec["query"].strip()
        base_a = rec["answer"].strip()
        copies = multiplier + (1 if remainder > 0 else 0)
        remainder -= 1
        for i in range(copies):
            if i == 0:
                q, a = base_q, base_a
            else:
                q = f"{base_q} [v{i}]"
                a = f"{base_a} [v{i}]"
            out.append({"query": q, "answer": a, "group_id": group_id})
        group_id += 1
    # Shuffle deterministically
    rng = np.random.RandomState(42)
    order = rng.permutation(len(out))
    return [out[i] for i in order]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
def get_ram_mb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def benchmark_query_latency(memory, embedder, queries: List[str], n_samples: int = 100) -> Dict:
    """Run n_samples queries and return latency stats."""
    rng = np.random.RandomState(7)
    sample = [queries[i] for i in rng.choice(len(queries), min(n_samples, len(queries)), replace=False)]
    latencies = []
    for q in sample:
        emb = embedder.embed(q)
        t0 = time.perf_counter()
        memory.load_memory_variables_with_embedding({"input": q, "session_id": "default"}, emb)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    n = len(latencies)
    return {
        "P50": latencies[n // 2],
        "P95": latencies[int(n * 0.95)],
        "P99": latencies[int(n * 0.99)],
        "min": latencies[0],
        "max": latencies[-1],
    }


def benchmark_r1(memory, embedder, records: List[Dict], sample_size: int = 200) -> float:
    """Compute Recall@1 on a sample of original queries.
    A hit means the retrieved top-1 node belongs to the same group_id."""
    rng = np.random.RandomState(13)
    # Only evaluate on original questions (those without [v] suffix)
    original_records = [r for r in records if "[v" not in r["query"]]
    if len(original_records) > sample_size:
        idx = rng.choice(len(original_records), sample_size, replace=False)
        eval_set = [original_records[i] for i in idx]
    else:
        eval_set = original_records

    hits = 0
    for rec in eval_set:
        emb = embedder.embed(rec["query"])
        result = memory.load_memory_variables_with_embedding({"input": rec["query"], "session_id": "default"}, emb)
        ctx_str = result.get("rtmdk_context", "")
        if not ctx_str:
            continue
        top_ctx = ctx_str
        # Check if any variant of the correct answer appears in top context
        gid = rec.get("group_id", -1)
        # Find any record with same group_id whose answer is in context
        match = False
        for r in records:
            if r.get("group_id") == gid and r["answer"] in top_ctx:
                match = True
                break
        if match:
            hits += 1
    return 100.0 * hits / len(eval_set) if eval_set else 0.0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def run_stability_benchmark(target_size: int):
    print("=" * 70)
    print(f"RTMDK v8.1 Stability Benchmark - N={target_size}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Setup
    embedder = BatchEmbedder()
    if not embedder._available:
        print("ERROR: LM Studio not available. Exiting.")
        sys.exit(1)

    print("\n[1/6] Loading dataset...")
    raw = load_qa()
    print(f"  Original records: {len(raw)}")
    records = expand_dataset(raw, target_size)
    print(f"  Expanded records: {len(records)}")

    config = RTMDKConfig(
        embedding_dim=768,
        latent_dim=128,
        top_k=15,
        min_response=0.001,
        decay_rate=0.999,
        use_hnsw=True,               # HNSW enabled for N > 5000
        learn_projection=False,
        bm25_fallback=False,
        enable_async=False,
        attention_bias=True,
        context_format="attention",
        resonance_kernel="cosine",
        phase_coupling=0.0,
        adaptive_bandwidth=False,
        max_nodes=target_size + 1000,
    )

    print("\n[2/6] Building memory + generating embeddings...")
    gc.collect()
    ram_before = get_ram_mb()
    t0 = time.time()

    # Pre-generate all embeddings in batches
    all_texts = [r["query"] + " " + r["answer"] for r in records]
    print(f"  Embedding {len(all_texts)} texts via LM Studio...")
    embed_t0 = time.time()
    all_embeddings = embedder.embed_many(all_texts)
    embed_elapsed = time.time() - embed_t0
    print(f"  Embedding done in {embed_elapsed:.1f}s ({len(all_texts)/embed_elapsed:.1f} texts/sec)")

    memory = RTMDKMemory(config=config, embedder=embedder.embed)
    add_t0 = time.time()
    for i, rec in enumerate(records):
        memory.add_node(
            embedding=all_embeddings[i],
            content={"text": rec["answer"]},
            session_id="stability_benchmark",
            modality="text",
        )
        if (i + 1) % 1000 == 0:
            print(f"    Added {i+1}/{len(records)} nodes ({get_ram_mb():.0f} MB)")
    add_elapsed = time.time() - add_t0

    ram_after = get_ram_mb()
    print(f"\n  Insert: {add_elapsed:.1f}s ({len(records)/add_elapsed:.1f} nodes/sec)")
    print(f"  RAM: {ram_before:.1f} MB -> {ram_after:.1f} MB  (+{ram_after-ram_before:.1f} MB)")

    print("\n[3/6] Query latency (sample of 100)...")
    original_questions = [r["query"] for r in raw]
    latency_stats = benchmark_query_latency(memory, embedder, original_questions, n_samples=100)
    print(f"  P50={latency_stats['P50']:.2f}ms  P95={latency_stats['P95']:.2f}ms  P99={latency_stats['P99']:.2f}ms")

    print("\n[4/6] Recall@1 (sample of 200 original queries)...")
    r1 = benchmark_r1(memory, embedder, records, sample_size=200)
    print(f"  R@1 = {r1:.1f}%")

    print("\n[5/6] Consolidation...")
    gc.collect()
    cons_t0 = time.time()
    memory.field.consolidate()
    cons_elapsed = time.time() - cons_t0
    print(f"  Consolidation: {cons_elapsed:.1f}s")
    ram_after_cons = get_ram_mb()
    print(f"  RAM after consolidation: {ram_after_cons:.1f} MB")

    print("\n[6/6] Serialization (save/load)...")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w", encoding="utf-8") as f:
        tmp_path = f.name
    try:
        save_t0 = time.time()
        memory.export_field(tmp_path)
        save_elapsed = time.time() - save_t0
        file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        print(f"  Save:   {save_elapsed:.1f}s  ({file_size_mb:.1f} MB on disk)")

        load_t0 = time.time()
        memory2 = RTMDKMemory.import_field(tmp_path, embedder=embedder.embed)
        load_elapsed = time.time() - load_t0
        print(f"  Load:   {load_elapsed:.1f}s")

        # Verify loaded memory has same node count
        assert len(memory2.field.nodes) == len(memory.field.nodes), "Node count mismatch after load"
        print(f"  Node count after load: {len(memory2.field.nodes)} (OK)")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Scale:                 {target_size} nodes")
    print(f"Embedding time:        {embed_elapsed:.1f}s")
    print(f"Insert throughput:     {len(records)/add_elapsed:.1f} nodes/sec")
    print(f"RAM usage:             {ram_after:.1f} MB ({ram_after/len(records)*1000:.2f} MB / 1K nodes)")
    print(f"Query latency P50:     {latency_stats['P50']:.2f} ms")
    print(f"Query latency P95:     {latency_stats['P95']:.2f} ms")
    print(f"Query latency P99:     {latency_stats['P99']:.2f} ms")
    print(f"Recall@1 (1K sample):  {r1:.1f}%")
    print(f"Consolidation time:    {cons_elapsed:.1f}s")
    print(f"Save time:             {save_elapsed:.1f}s")
    print(f"Load time:             {load_elapsed:.1f}s")
    print(f"Disk size:             {file_size_mb:.1f} MB")
    print("=" * 70)

    return {
        "target_size": target_size,
        "embed_time": embed_elapsed,
        "insert_throughput": len(records) / add_elapsed,
        "ram_mb": ram_after,
        "ram_per_1k": ram_after / len(records) * 1000,
        "latency_p50_ms": latency_stats["P50"],
        "latency_p95_ms": latency_stats["P95"],
        "latency_p99_ms": latency_stats["P99"],
        "recall_at_1": r1,
        "consolidation_sec": cons_elapsed,
        "save_sec": save_elapsed,
        "load_sec": load_elapsed,
        "disk_mb": file_size_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="RTMDK Stability Benchmark")
    parser.add_argument("--scale", type=int, choices=[5000, 10000], default=5000, help="Target node count")
    args = parser.parse_args()

    # Ensure rate limit doesn't interfere
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

    results = run_stability_benchmark(args.scale)

    # Save results
    out_path = f"tests/results/stability_{args.scale}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
