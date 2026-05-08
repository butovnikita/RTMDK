"""Head-to-head benchmark: RTMDK vs Cosine RAG baseline.

Measures:
  - Recall@K (is ground-truth context in top-K?)
  - Latency (ms per query)
  - MRR (Mean Reciprocal Rank)

Usage:
    python scripts/bench_rag_vs_rtmdk.py --dataset datasets/qa_1000_en.json --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dataset(path: str, n: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    return records[:n]


def make_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed(text: str):
        return model.encode(text, convert_to_numpy=True).astype(np.float32)
    return embed


def bench_cosine_rag(records: List[Dict], embedder, top_k: int = 5) -> Dict[str, float]:
    contexts = [r["context"] for r in records]
    queries = [r["query"] for r in records]
    ctx_embs = np.vstack([embedder(c) for c in contexts])
    ctx_embs = ctx_embs / (np.linalg.norm(ctx_embs, axis=1, keepdims=True) + 1e-12)

    recalls = []
    latencies = []
    ranks = []
    for i, query in enumerate(queries):
        q_emb = embedder(query)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        t0 = time.perf_counter()
        sims = ctx_embs @ q_emb
        top_idx = np.argsort(-sims)[:top_k]
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(1 if i in top_idx else 0)
        rank = np.where(np.argsort(-sims) == i)[0]
        ranks.append(1.0 / (rank[0] + 1) if len(rank) else 0.0)
    return {
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(ranks)),
        "latency_p50_ms": float(np.median(latencies)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def bench_rtmdk(records: List[Dict], embedder, top_k: int = 5, latent_dim: int = 64) -> Dict[str, float]:
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
    from rtmdk.memory.config import RTMDKConfig
    from rtmdk.memory.core import RTMDKMemory

    cfg = RTMDKConfig(
        latent_dim=latent_dim,
        embedding_dim=384,
        max_nodes=len(records) + 10,
        top_k=top_k,
        min_response=0.001,
        bandwidth=1.0,
        phase_coupling=0.3,
        use_hnsw=True,
        learn_projection=False,
    )
    memory = RTMDKMemory(config=cfg, embedder=embedder)
    for rec in records:
        emb = embedder(rec["context"])
        memory.add_node(embedding=emb, content={"text": rec["context"]})

    queries = [r["query"] for r in records]
    recalls = []
    latencies = []
    ranks = []
    node_ids = {n.id: idx for idx, n in enumerate(memory.field.nodes.values())}
    for i, query in enumerate(queries):
        t0 = time.perf_counter()
        results = memory.field.query(embedder(query), top_k=top_k * 2)
        top_idx = [node_ids.get(nid, -1) for nid, _, _ in results[:top_k]]
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(1 if i in top_idx else 0)
        rank = None
        for rank_pos, (nid, _, _) in enumerate(results):
            if node_ids.get(nid) == i:
                rank = rank_pos
                break
        ranks.append(1.0 / (rank + 1) if rank is not None else 0.0)
    return {
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(ranks)),
        "latency_p50_ms": float(np.median(latencies)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/qa_1000_en.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print("Loading dataset...")
    records = load_dataset(args.dataset, args.n)
    print(f"Records: {len(records)}")

    print("Loading embedder (all-MiniLM-L6-v2)...")
    embedder = make_embedder()

    print("\n=== Cosine RAG Baseline ===")
    cosine = bench_cosine_rag(records, embedder, top_k=args.top_k)
    for k, v in cosine.items():
        print(f"  {k}: {v:.4f}")

    for latent_dim in (384, 256, 128, 64):
        print(f"\n=== RTMDK Resonance (latent_dim={latent_dim}, identity projection) ===")
        rtmdk = bench_rtmdk(records, embedder, top_k=args.top_k, latent_dim=latent_dim)
        for k, v in rtmdk.items():
            print(f"  {k}: {v:.4f}")
        print(f"  vs RAG — recall delta={rtmdk['recall_at_k']-cosine['recall_at_k']:+.3f}, mrr delta={rtmdk['mrr']-cosine['mrr']:+.3f}")


if __name__ == "__main__":
    main()
