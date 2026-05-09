#!/usr/bin/env python3
"""BEIR-style benchmark: RTMDK vs Cosine on real embeddings.

Usage:
    python scripts/bench_beir_rtmdk.py --dataset scifact --model BAAI/bge-small-en-v1.5
    python scripts/bench_beir_rtmdk.py --dataset nfcorpus --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def load_beir_dataset(name: str):
    """Load a BEIR dataset (downloads on first run)."""
    try:
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:
        raise RuntimeError("pip install beir") from exc
    data_path = f"./beir_datasets/{name}"
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def embed_corpus_and_queries(corpus, queries, model_name: str, batch_size: int = 32):
    """Embed all passages and queries with sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"Embedding {len(corpus_texts)} passages + {len(query_texts)} queries with {model_name}...")
    t0 = time.perf_counter()
    corpus_embs = model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True)
    query_embs = model.encode(query_texts, batch_size=batch_size, show_progress_bar=True)
    print(f"Embedding done in {time.perf_counter() - t0:.1f}s")
    return corpus_ids, corpus_embs, query_ids, query_embs


def build_rtmdk(corpus_ids: List[str], corpus_embs: np.ndarray) -> RTMDKMemory:
    """Build RTMDK memory from pre-computed embeddings."""
    dim = corpus_embs.shape[1]
    cfg = RTMDKConfig(
        latent_dim=dim,
        embedding_dim=dim,
        max_nodes=len(corpus_ids) + 100,
        top_k=10,
        min_response=0.001,
        bandwidth=1.0,
        phase_coupling=0.0,
        use_hnsw=True,
        hnsw_min_nodes=50,
        bm25_fallback=False,
        learn_projection=False,
        projection_mode="identity",
        pipeline_enabled=True,
    )
    mem = RTMDKMemory(config=cfg, embedder=lambda t: np.zeros(dim, dtype=np.float32))
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

    print("Ingesting into RTMDK...")
    t0 = time.perf_counter()
    for cid, emb in zip(corpus_ids, corpus_embs):
        mem.add_node(content={"text": cid}, embedding=emb.astype(np.float32))
    print(f"Ingest done in {time.perf_counter() - t0:.1f}s")
    return mem


def evaluate_rtmdk(mem: RTMDKMemory, query_ids: List[str], query_embs: np.ndarray,
                   qrels: Dict, corpus_ids: List[str], top_k: int = 10) -> Dict[str, float]:
    """Run queries and compute recall@k and MRR."""
    recalls = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr_sum = 0.0
    n = 0
    latencies = []

    for qid, qemb in zip(query_ids, query_embs):
        if qid not in qrels:
            continue
        relevant = set(qrels[qid].keys())
        if not relevant:
            continue

        t0 = time.perf_counter()
        results = mem.retrieve_nodes_pipeline("", top_k=top_k)
        # Fallback: direct field query with embedding
        if not results.get("nodes"):
            results = mem.field.query(qemb.astype(np.float32), top_k=top_k)
            result_ids = [r[0] for r in results]
        else:
            result_ids = [n.id for n in results["nodes"]]
        latencies.append((time.perf_counter() - t0) * 1000)

        # Recall@k
        for k in recalls:
            hits = len(relevant.intersection(result_ids[:k]))
            recalls[k] += hits / len(relevant)

        # MRR
        for rank, rid in enumerate(result_ids, 1):
            if rid in relevant:
                mrr_sum += 1.0 / rank
                break
        n += 1

    if n == 0:
        return {}
    return {
        "recall@1": recalls[1] / n,
        "recall@5": recalls[5] / n,
        "recall@10": recalls[10] / n,
        "mrr": mrr_sum / n,
        "queries": n,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def evaluate_cosine(corpus_embs: np.ndarray, query_embs: np.ndarray,
                    query_ids: List[str], qrels: Dict, corpus_ids: List[str],
                    top_k: int = 10) -> Dict[str, float]:
    """Pure cosine similarity baseline."""
    # Normalize
    corpus_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-8)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)

    recalls = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr_sum = 0.0
    n = 0
    latencies = []

    for qid, qemb in zip(query_ids, query_norm):
        if qid not in qrels:
            continue
        relevant = set(qrels[qid].keys())
        if not relevant:
            continue

        t0 = time.perf_counter()
        scores = corpus_norm @ qemb
        top_idx = np.argsort(scores)[::-1][:top_k]
        result_ids = [corpus_ids[i] for i in top_idx]
        latencies.append((time.perf_counter() - t0) * 1000)

        for k in recalls:
            hits = len(relevant.intersection(result_ids[:k]))
            recalls[k] += hits / len(relevant)

        for rank, rid in enumerate(result_ids, 1):
            if rid in relevant:
                mrr_sum += 1.0 / rank
                break
        n += 1

    if n == 0:
        return {}
    return {
        "recall@1": recalls[1] / n,
        "recall@5": recalls[5] / n,
        "recall@10": recalls[10] / n,
        "mrr": mrr_sum / n,
        "queries": n,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def main():
    parser = argparse.ArgumentParser(description="BEIR benchmark for RTMDK")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Sentence-transformers model")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    corpus, queries, qrels = load_beir_dataset(args.dataset)
    corpus_ids, corpus_embs, query_ids, query_embs = embed_corpus_and_queries(
        corpus, queries, args.model, args.batch_size)

    # Cosine baseline
    print("\n--- Cosine Baseline ---")
    cosine_metrics = evaluate_cosine(
        corpus_embs, query_embs, query_ids, qrels, corpus_ids, args.top_k)
    for k, v in cosine_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # RTMDK
    print("\n--- RTMDK ---")
    mem = build_rtmdk(corpus_ids, corpus_embs)
    rtmdk_metrics = evaluate_rtmdk(
        mem, query_ids, query_embs, qrels, corpus_ids, args.top_k)
    for k, v in rtmdk_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Summary
    print("\n--- Comparison ---")
    for k in ["recall@1", "recall@5", "recall@10", "mrr"]:
        c = cosine_metrics.get(k, 0)
        r = rtmdk_metrics.get(k, 0)
        delta = ((r / c) - 1) * 100 if c > 0 else 0
        print(f"  {k}: Cosine={c:.4f}  RTMDK={r:.4f}  delta={delta:+.1f}%")


if __name__ == "__main__":
    main()
