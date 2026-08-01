#!/usr/bin/env python3
"""BEIR-style benchmark: RTMDK vs Cosine vs FAISS on real embeddings.

Usage:
    python scripts/bench_beir_rtmdk.py --datasets scifact nfcorpus --model sentence-transformers/all-MiniLM-L6-v2
    python scripts/bench_beir_rtmdk.py --dataset fiqa --model BAAI/bge-small-en-v1.5 --rtmdk-bandwidth 0.8 --rtmdk-phase-coupling 0.2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def load_beir_dataset(name: str):
    """Load a BEIR dataset (downloads on first run)."""
    try:
        from beir.datasets.data_loader import GenericDataLoader
        from beir import util
    except ImportError as exc:
        raise RuntimeError("pip install beir") from exc
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    data_path = f"./beir_datasets/{name}"
    if not os.path.exists(os.path.join(data_path, "corpus.jsonl")):
        print(f"Downloading BEIR dataset '{name}'...")
        util.download_and_unzip(url, "./beir_datasets")
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
    corpus_embs = model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    query_embs = model.encode(query_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    print(f"Embedding done in {time.perf_counter() - t0:.1f}s")
    return corpus_ids, corpus_embs.astype(np.float32), query_ids, query_embs.astype(np.float32)


def build_rtmdk(
    corpus_ids: List[str],
    corpus_embs: np.ndarray,
    bandwidth: float = 1.0,
    phase_coupling: float = 0.0,
    hyperbolic: bool = False,
    hnsw_min_nodes: int = 50,
) -> RTMDKMemory:
    """Build RTMDK memory from pre-computed embeddings (batched insert)."""
    dim = corpus_embs.shape[1]
    cfg = RTMDKConfig(
        latent_dim=dim,
        embedding_dim=dim,
        max_nodes=len(corpus_ids) + 100,
        top_k=10,
        min_response=0.001,
        bandwidth=bandwidth,
        phase_coupling=phase_coupling,
        hyperbolic=hyperbolic,
        use_hnsw=True,
        hnsw_min_nodes=hnsw_min_nodes,
        bm25_fallback=False,
        learn_projection=False,
        projection_mode="identity",
        pipeline_enabled=True,
    )
    mem = RTMDKMemory(config=cfg, embedder=lambda t: np.zeros(dim, dtype=np.float32))
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

    print("Ingesting into RTMDK (batched)...")
    t0 = time.perf_counter()
    batch_size = 1000
    for i in range(0, len(corpus_ids), batch_size):
        batch_ids = corpus_ids[i : i + batch_size]
        batch_embs = corpus_embs[i : i + batch_size]
        mem.add_nodes_batch(
            embeddings=batch_embs,
            contents=[{"text": cid} for cid in batch_ids],
            node_ids=batch_ids,
        )
    print(f"Ingest done in {time.perf_counter() - t0:.1f}s")
    return mem


def _ndcg_at_k(result_ids: List[str], qrels_dict: Dict[str, str], k: int) -> float:
    """Compute nDCG@k for a single query."""
    dcg = 0.0
    for i, rid in enumerate(result_ids[:k], 1):
        rel = float(qrels_dict.get(rid, 0))
        dcg += (2**rel - 1) / np.log2(i + 1)
    # Ideal DCG
    ideal_rels = sorted([float(v) for v in qrels_dict.values()], reverse=True)
    idcg = sum((2**r - 1) / np.log2(i + 1) for i, r in enumerate(ideal_rels[:k], 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_rtmdk(
    mem: RTMDKMemory, query_ids: List[str], query_embs: np.ndarray, qrels: Dict, corpus_ids: List[str], top_k: int = 10
) -> Dict[str, float]:
    """Run queries and compute recall@k, MRR, nDCG."""
    recalls = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr_sum = 0.0
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
    n = 0
    latencies = []

    for qid, qemb in zip(query_ids, query_embs):
        if qid not in qrels:
            continue
        relevant = set(qrels[qid].keys())
        if not relevant:
            continue

        t0 = time.perf_counter()
        results = mem.field.query(qemb, top_k=top_k)
        result_ids = [r[0] for r in results]
        latencies.append((time.perf_counter() - t0) * 1000)

        for k in recalls:
            hits = len(relevant.intersection(result_ids[:k]))
            recalls[k] += hits / len(relevant)

        for rank, rid in enumerate(result_ids, 1):
            if rid in relevant:
                mrr_sum += 1.0 / rank
                break

        ndcg5_sum += _ndcg_at_k(result_ids, qrels[qid], 5)
        ndcg10_sum += _ndcg_at_k(result_ids, qrels[qid], 10)
        n += 1

    if n == 0:
        return {}
    return {
        "recall@1": recalls[1] / n,
        "recall@5": recalls[5] / n,
        "recall@10": recalls[10] / n,
        "mrr": mrr_sum / n,
        "nDCG@5": ndcg5_sum / n,
        "nDCG@10": ndcg10_sum / n,
        "queries": n,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def evaluate_cosine(
    corpus_embs: np.ndarray,
    query_embs: np.ndarray,
    query_ids: List[str],
    qrels: Dict,
    corpus_ids: List[str],
    top_k: int = 10,
) -> Dict[str, float]:
    """Pure cosine similarity baseline."""
    corpus_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-8)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)

    recalls = {1: 0.0, 5: 0.0, 10: 0.0}
    mrr_sum = 0.0
    ndcg5_sum = 0.0
    ndcg10_sum = 0.0
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

        ndcg5_sum += _ndcg_at_k(result_ids, qrels[qid], 5)
        ndcg10_sum += _ndcg_at_k(result_ids, qrels[qid], 10)
        n += 1

    if n == 0:
        return {}
    return {
        "recall@1": recalls[1] / n,
        "recall@5": recalls[5] / n,
        "recall@10": recalls[10] / n,
        "mrr": mrr_sum / n,
        "nDCG@5": ndcg5_sum / n,
        "nDCG@10": ndcg10_sum / n,
        "queries": n,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def evaluate_faiss(
    corpus_embs: np.ndarray,
    query_embs: np.ndarray,
    query_ids: List[str],
    qrels: Dict,
    corpus_ids: List[str],
    top_k: int = 10,
) -> Optional[Dict[str, float]]:
    """FAISS IVF/Flat baseline (if available)."""
    try:
        import faiss
    except ImportError:
        return None

    dim = corpus_embs.shape[1]
    nlist = min(64, len(corpus_embs) // 10)
    if nlist < 4:
        nlist = 1
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    corpus_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-8)
    query_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
    index.train(corpus_norm)
    index.add(corpus_norm)
    index.nprobe = 10

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
        scores, indices = index.search(qemb.reshape(1, -1), top_k)
        result_ids = [corpus_ids[i] for i in indices[0]]
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


def run_single_dataset(
    dataset_name: str, model_name: str, batch_size: int, rtmdk_cfg: Dict, top_k: int
) -> Dict[str, Dict]:
    corpus, queries, qrels = load_beir_dataset(dataset_name)
    corpus_ids, corpus_embs, query_ids, query_embs = embed_corpus_and_queries(corpus, queries, model_name, batch_size)

    results = {}

    # Cosine
    print("\n--- Cosine Baseline ---")
    results["cosine"] = evaluate_cosine(corpus_embs, query_embs, query_ids, qrels, corpus_ids, top_k)
    for k, v in results["cosine"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # FAISS
    faiss_metrics = evaluate_faiss(corpus_embs, query_embs, query_ids, qrels, corpus_ids, top_k)
    if faiss_metrics:
        print("\n--- FAISS IVF ---")
        results["faiss"] = faiss_metrics
        for k, v in faiss_metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # RTMDK
    print("\n--- RTMDK ---")
    mem = build_rtmdk(corpus_ids, corpus_embs, **rtmdk_cfg)
    results["rtmdk"] = evaluate_rtmdk(mem, query_ids, query_embs, qrels, corpus_ids, top_k)
    for k, v in results["rtmdk"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return results


def main():
    parser = argparse.ArgumentParser(description="BEIR benchmark for RTMDK")
    parser.add_argument("--datasets", nargs="+", default=["scifact"], help="BEIR dataset names")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="Sentence-transformers model")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rtmdk-bandwidth", type=float, default=1.0)
    parser.add_argument("--rtmdk-phase-coupling", type=float, default=0.0)
    parser.add_argument("--rtmdk-hyperbolic", action="store_true")
    parser.add_argument("--rtmdk-hnsw-min-nodes", type=int, default=50)
    args = parser.parse_args()

    rtmdk_cfg = {
        "bandwidth": args.rtmdk_bandwidth,
        "phase_coupling": args.rtmdk_phase_coupling,
        "hyperbolic": args.rtmdk_hyperbolic,
        "hnsw_min_nodes": args.rtmdk_hnsw_min_nodes,
    }

    all_results: Dict[str, Dict[str, Dict]] = {}
    for dataset in args.datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset}")
        print(f"{'='*70}")
        all_results[dataset] = run_single_dataset(dataset, args.model, args.batch_size, rtmdk_cfg, args.top_k)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    headers = ["dataset", "method", "recall@1", "recall@5", "recall@10", "nDCG@5", "nDCG@10", "mrr", "p50_ms", "p99_ms"]
    print("  ".join(f"{h:>12}" for h in headers))
    for dataset, methods in all_results.items():
        for method, metrics in methods.items():
            row = [
                dataset,
                method,
                f"{metrics.get('recall@1', 0):.4f}",
                f"{metrics.get('recall@5', 0):.4f}",
                f"{metrics.get('recall@10', 0):.4f}",
                f"{metrics.get('nDCG@5', 0):.4f}",
                f"{metrics.get('nDCG@10', 0):.4f}",
                f"{metrics.get('mrr', 0):.4f}",
                f"{metrics.get('latency_p50_ms', 0):.2f}",
                f"{metrics.get('latency_p99_ms', 0):.2f}",
            ]
            print("  ".join(f"{c:>12}" for c in row))


if __name__ == "__main__":
    main()
