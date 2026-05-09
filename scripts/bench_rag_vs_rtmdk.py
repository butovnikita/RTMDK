"""Head-to-head benchmark: RTMDK vs Cosine RAG baseline.

Modes:
  sbert   — sentence-transformers all-MiniLM-L6-v2 (default)
  sot     — Self-Organizing Tokenizer (bootstrapped from SBERT)
  bgem3   — BGE-M3 dense + sparse + multi-vector
  all     — run all three sequentially

Measures:
  - Recall@K (is ground-truth context in top-K?)
  - Latency (ms per query)
  - MRR (Mean Reciprocal Rank)

Usage:
    python scripts/bench_rag_vs_rtmdk.py --mode all --dataset datasets/qa_1000_en.json --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dataset(path: str, n: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    return records[:n]


def make_sbert_embedder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed(text: str):
        return model.encode(text, convert_to_numpy=True).astype(np.float32)
    return embed


def bench_cosine_rag(records: List[Dict], embedder: Callable, top_k: int = 5) -> Dict[str, float]:
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


def bench_rtmdk(
    records: List[Dict],
    *,
    mode: str = "sbert",
    top_k: int = 5,
    latent_dim: int = 384,
) -> Dict[str, float]:
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
    from rtmdk.memory.config import RTMDKConfig
    from rtmdk.memory.core import RTMDKMemory

    # Base benchmark config
    cfg = RTMDKConfig.benchmark() if latent_dim == 384 else RTMDKConfig(
        latent_dim=latent_dim,
        embedding_dim=384,
        max_nodes=len(records) + 10,
        top_k=top_k,
        min_response=0.001,
        bandwidth=1.0,
        phase_coupling=0.3,
        use_hnsw=True,
        learn_projection=False,
        projection_mode="identity",
    )

    if mode == "sot":
        cfg.sot_enabled = True
        cfg.sot_use_for_query = True
        cfg.sot_subword_seed = True
        cfg.sot_attention_pooling = True
        cfg.sot_max_vocab = 10000
        cfg.sot_tokenization_mode = "word"
    elif mode == "sot_v2":
        cfg.sot_enabled = False  # Use v2 instead of v1
    elif mode == "bgem3":
        cfg.bgem3_enabled = True

    # Create memory
    if mode == "sbert":
        embedder = make_sbert_embedder()
        memory = RTMDKMemory(config=cfg, embedder=embedder)
    elif mode == "sot":
        # SOT does not need an external embedder after bootstrap, but we pass
        # a dummy to satisfy RTMDKMemory constructor; we will override usage.
        memory = RTMDKMemory(config=cfg, embedder=lambda t: np.zeros(cfg.latent_dim, dtype=np.float32))
    elif mode == "sot_v2":
        from rtmdk.memory.sot_v2.integration import SOTv2Embedder
        corpus = [r["context"] + " " + r["query"] for r in records]
        sot_v2 = SOTv2Embedder(latent_dim=latent_dim)
        sot_v2.train(corpus)
        memory = RTMDKMemory(config=cfg, embedder=sot_v2)
    elif mode == "bgem3":
        memory = RTMDKMemory(config=cfg, embedder=make_sbert_embedder())
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Ingest ---
    if mode == "sot":
        print("  Bootstrapping SOT from SBERT teacher...")
        texts = [r["context"] + " " + r["query"] for r in records]
        memory.field.sot_bootstrap(texts, teacher_model="all-MiniLM-L6-v2",
                                   fit_projection_only=False, n_epochs=50)
        def sot_embed(text: str):
            tokens = memory.field._projection_mgr.sot_tokenizer.encode(text)
            return memory.field._projection_mgr.sot_tokenizer.embed(tokens).astype(np.float32)
        for rec in records:
            emb = sot_embed(rec["context"])
            memory.add_node(embedding=emb, content={"text": rec["context"]}, phase=0.0)
        query_embs = [sot_embed(rec["query"]) for rec in records]
        query_sparse = [None] * len(records)
    elif mode == "bgem3":
        bgem3 = memory.bgem3_embedder
        if bgem3 is None:
            raise RuntimeError("BGE-M3 embedder not available")
        print("  Using BGE-M3 embedder (dense + sparse)")
        ctx_bgems = bgem3.encode([rec["context"] for rec in records])
        for rec, bge in zip(records, ctx_bgems):
            memory.add_node(
                embedding=bge.dense,
                content={"text": rec["context"], "sparse_embedding": bge.sparse},
            )
        query_bgems = bgem3.encode([rec["query"] for rec in records])
        query_embs = [b.dense for b in query_bgems]
        query_sparse = [b.sparse for b in query_bgems]
    else:  # sbert
        mem_embedder = memory.embedder
        for rec in records:
            emb = mem_embedder(rec["context"])
            memory.add_node(embedding=emb, content={"text": rec["context"]})
        query_embs = [mem_embedder(rec["query"]) for rec in records]
        query_sparse = [None] * len(records)

    queries = [r["query"] for r in records]
    node_ids = {n.id: idx for idx, n in enumerate(memory.field.nodes.values())}

    # --- Single-query latency (full pipeline) ---
    recalls = []
    latencies = []
    ranks = []
    for i, (query, q_emb, q_sparse) in enumerate(zip(queries, query_embs, query_sparse)):
        t0 = time.perf_counter()
        results = memory.retrieve_nodes(query, q_emb, top_k=top_k * 2, sparse_vec=q_sparse)
        top_idx = [node_ids.get(nid, -1) for nid, _, _ in results[:top_k]]
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(1 if i in top_idx else 0)
        rank = None
        for rank_pos, (nid, _, _) in enumerate(results):
            if node_ids.get(nid) == i:
                rank = rank_pos
                break
        ranks.append(1.0 / (rank + 1) if rank is not None else 0.0)

    # --- Batch-query latency ---
    batch_latencies = []
    batch_size = 32
    for offset in range(0, len(query_embs), batch_size):
        batch = query_embs[offset:offset + batch_size]
        t0 = time.perf_counter()
        _ = memory.batch_query(batch, top_k=top_k * 2)
        batch_latencies.append((time.perf_counter() - t0) * 1000)
    per_query_batch_latency = [lat / batch_size for lat in batch_latencies for _ in range(batch_size)][:len(query_embs)]

    return {
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(ranks)),
        "latency_p50_ms": float(np.median(latencies)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "batch_latency_p50_ms": float(np.median(per_query_batch_latency)) if per_query_batch_latency else 0.0,
    }


def run_single(mode: str, records: List[Dict], top_k: int, latent_dim: int, cosine: Dict[str, float]) -> Dict[str, float]:
    print(f"\n=== RTMDK ({mode.upper()}, latent_dim={latent_dim}) ===")
    rtmdk = bench_rtmdk(records, mode=mode, top_k=top_k, latent_dim=latent_dim)
    for k, v in rtmdk.items():
        print(f"  {k}: {v:.4f}")
    print(f"  vs RAG — recall delta={rtmdk['recall_at_k']-cosine['recall_at_k']:+.3f}, mrr delta={rtmdk['mrr']-cosine['mrr']:+.3f}")
    if "batch_latency_p50_ms" in rtmdk:
        print(f"  batch per-query latency p50: {rtmdk['batch_latency_p50_ms']:.4f} ms")
    return rtmdk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/qa_1000_en.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--mode", default="sbert", choices=["sbert", "sot", "sot_v2", "bgem3", "all"])
    parser.add_argument("--latent_dim", type=int, default=384, help="Only used when mode != all")
    args = parser.parse_args()

    print("Loading dataset...")
    records = load_dataset(args.dataset, args.n)
    print(f"Records: {len(records)}")

    print("\n=== Cosine RAG Baseline (SBERT) ===")
    embedder = make_sbert_embedder()
    cosine = bench_cosine_rag(records, embedder, top_k=args.top_k)
    for k, v in cosine.items():
        print(f"  {k}: {v:.4f}")

    if args.mode == "all":
        for mode in ("sbert", "sot", "sot_v2", "bgem3"):
            latent_dim = 384
            try:
                run_single(mode, records, args.top_k, latent_dim, cosine)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        run_single(args.mode, records, args.top_k, args.latent_dim, cosine)


if __name__ == "__main__":
    main()
