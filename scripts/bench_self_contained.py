"""Fully self-contained benchmark: SOT v2 + RTMDK, no external embedders.

This script uses ONLY RTMDK's built-in SOT v2 embedder and BM25 index.
No sentence-transformers, no FAISS, no PyTorch at inference time.

Datasets:
- qa_1000_en.json
- comprehensive_500.json (EN + RU subsets)
- ms_marco_dev.json
- rubq.json
"""

import os

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

import json
import time
import argparse
from typing import List, Dict
from pathlib import Path

import numpy as np

from rtmdk.memory.sot_v2.integration import SOTv2Embedder, _word_tokenize
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory

DATASET_PATHS = {
    "qa_1000_en": "datasets/qa_1000_en.json",
    "comprehensive_500": "datasets/comprehensive_500.json",
    "ms_marco": "datasets/ms_marco_dev.json",
    "rubq": "datasets/rubq.json",
    "sts": "datasets/sts_benchmark.json",
}


def load_dataset(path: str, language: str = None, max_records: int = None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data)
    if language:
        records = [r for r in records if r.get("language") == language]
    if max_records:
        records = records[:max_records]
    return records


def dataset_hardness(records: List[Dict]) -> dict:
    """Heuristics to predict if contrastive fine-tuning will help.

    Returns dict with:
        - queries_per_context: avg queries sharing same context
        - vocab_overlap_qc: mean Jaccard overlap between query and context vocab
        - lexical_diversity: unique words / total words in queries
        - recommend_contrastive: bool
    """
    from collections import Counter

    ctx_counts = Counter(r["context"] for r in records)
    qpc = sum(ctx_counts.values()) / len(ctx_counts)

    # Vocab overlap
    overlaps = []
    for r in records:
        q_words = set(_word_tokenize(r["query"]))
        c_words = set(_word_tokenize(r["context"]))
        if q_words or c_words:
            overlaps.append(len(q_words & c_words) / len(q_words | c_words))
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0

    # Lexical diversity
    all_query_words = []
    for r in records:
        all_query_words.extend(_word_tokenize(r["query"]))
    diversity = len(set(all_query_words)) / len(all_query_words) if all_query_words else 0

    # Heuristic: contrastive helps when many queries per context AND low vocab overlap
    recommend = (qpc > 2.0) and (avg_overlap < 0.4)

    return {
        "queries_per_context": qpc,
        "vocab_overlap_qc": avg_overlap,
        "lexical_diversity": diversity,
        "recommend_contrastive": recommend,
    }


def build_sot_embedder(records: List[Dict], contrastive: bool = False, latent_dim: int = 384, model_path: str = None):
    """Train SOT v2 on corpus + queries. Optionally contrastive-fine-tune."""
    if model_path and Path(model_path).exists():
        print(f"  Loading SOT v2 from {model_path}")
        return SOTv2Embedder.load(model_path)

    corpus_texts = list({r["context"] for r in records})
    queries = [r["query"] for r in records]
    all_texts = corpus_texts + queries

    sot = SOTv2Embedder(latent_dim=latent_dim, window_size=5, a=0.01, remove_pc=True)
    sot.train(all_texts)

    if contrastive:
        tokenized_queries = []
        tokenized_positives = []
        for rec in records:
            q_tok = [sot._vocab[w] for w in _word_tokenize(rec["query"]) if w in sot._vocab]
            c_tok = [sot._vocab[w] for w in _word_tokenize(rec["context"]) if w in sot._vocab]
            if q_tok and c_tok:
                tokenized_queries.append(q_tok)
                tokenized_positives.append(c_tok)

        if len(tokenized_queries) >= 10:
            sot._embedder.contrastive_fine_tune(
                tokenized_queries,
                tokenized_positives,
                n_epochs=5,
                lr=0.005,
                temperature=1.0,
                n_negatives=5,
            )

    if model_path:
        sot.save(model_path)
        print(f"  Saved SOT v2 to {model_path}")

    return sot


def evaluate_pure_cosine(doc_embs: np.ndarray, records: List[Dict], embedder, corpus_texts: List[str], top_k: int = 5):
    query_times = []
    hits = 0
    hits1 = 0
    ranks = []

    for rec in records:
        q_emb = embedder(rec["query"])
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        t0 = time.perf_counter()
        sims = doc_embs @ q_emb
        top_idx = np.argsort(-sims)[:top_k]
        query_times.append(time.perf_counter() - t0)

        target_idx = corpus_texts.index(rec["context"])
        if target_idx in top_idx:
            hits += 1
            rank = np.where(top_idx == target_idx)[0][0] + 1
            ranks.append(1.0 / rank)
            if rank == 1:
                hits1 += 1
        else:
            ranks.append(0.0)

    return {
        f"recall@{top_k}": hits / len(records),
        "recall@1": hits1 / len(records),
        "mrr": sum(ranks) / len(ranks),
        "latency_p50_ms": float(np.percentile(query_times, 50) * 1000),
    }


def evaluate_bm25_sot_hybrid(
    records: List[Dict], embedder, corpus_texts: List[str], alpha: float = 0.7, top_k: int = 5
):
    """BM25 + SOT cosine hybrid scoring.

    score = alpha * cosine_normalized + (1-alpha) * bm25_normalized
    """
    from rtmdk.support.bm25 import BM25Index

    # Build BM25 index on corpus
    bm25 = BM25Index()
    for i, ctx in enumerate(corpus_texts):
        bm25.add_document(str(i), ctx)

    # Precompute SOT doc embeddings
    doc_embs = np.vstack([embedder(t) for t in corpus_texts])
    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)

    query_times = []
    hits = 0
    hits1 = 0
    ranks = []

    for rec in records:
        q_text = rec["query"]
        target = rec["context"]
        target_idx = corpus_texts.index(target)

        t0 = time.perf_counter()
        q_emb = embedder(q_text)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

        # Cosine scores
        cos_sims = doc_embs @ q_emb

        # BM25 scores
        bm25_hits = bm25.search(q_text, len(corpus_texts))
        bm25_scores = np.zeros(len(corpus_texts), dtype=np.float32)
        for doc_id, score in bm25_hits:
            bm25_scores[int(doc_id)] = score

        # Normalize both to [0, 1]
        cos_norm = (cos_sims - cos_sims.min()) / (cos_sims.max() - cos_sims.min() + 1e-8)
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)

        # Hybrid
        hybrid = alpha * cos_norm + (1 - alpha) * bm25_norm
        top_idx = np.argsort(-hybrid)[:top_k]
        query_times.append(time.perf_counter() - t0)

        if target_idx in top_idx:
            hits += 1
            rank = np.where(top_idx == target_idx)[0][0] + 1
            ranks.append(1.0 / rank)
            if rank == 1:
                hits1 += 1
        else:
            ranks.append(0.0)

    return {
        f"recall@{top_k}": hits / len(records),
        "recall@1": hits1 / len(records),
        "mrr": sum(ranks) / len(ranks),
        "latency_p50_ms": float(np.percentile(query_times, 50) * 1000),
    }


def evaluate_rtmdk(memory, records: List[Dict], embedder, top_k: int = 5):
    hits = 0
    hits1 = 0
    ranks = []
    query_times = []

    for rec in records:
        q_text = rec["query"]
        target = rec["context"]
        q_emb = embedder(q_text)

        t0 = time.perf_counter()
        results = memory.retrieve_nodes(q_text, q_emb, top_k=top_k)
        query_times.append(time.perf_counter() - t0)

        contexts = [r[2].content.get("text", "") for r in results]
        if target in contexts:
            hits += 1
            rank = contexts.index(target) + 1
            ranks.append(1.0 / rank)
            if rank == 1:
                hits1 += 1
        else:
            ranks.append(0.0)

    return {
        f"recall@{top_k}": hits / len(records),
        "recall@1": hits1 / len(records),
        "mrr": sum(ranks) / len(ranks),
        "latency_p50_ms": float(np.percentile(query_times, 50) * 1000),
    }


def benchmark_dataset(name: str, records: List[Dict], contrastive: bool = False, save_model: str = None):
    corpus_texts = list({r["context"] for r in records})
    print(f"\n{'='*60}")
    print(f"Dataset: {name} | Records: {len(records)} | Contexts: {len(corpus_texts)}")
    print("=" * 60)

    hardness = dataset_hardness(records)
    print(
        f"  Hardness: qpc={hardness['queries_per_context']:.1f}, "
        f"overlap={hardness['vocab_overlap_qc']:.2f}, "
        f"diversity={hardness['lexical_diversity']:.2f}, "
        f"recommend_contrastive={hardness['recommend_contrastive']}"
    )

    # Train SOT v2
    model_path = f"models/sotv2_{name.replace(' ', '_')}.json" if save_model else None
    if model_path:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    embedder = build_sot_embedder(records, contrastive=contrastive, model_path=model_path)
    train_time = time.time() - t0
    print(f"SOT v2 training: {train_time:.2f}s")

    # Precompute doc embeddings for cosine
    doc_embs = np.vstack([embedder(t) for t in corpus_texts])
    doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)

    # --- Pure SOT cosine ---
    res_cos = evaluate_pure_cosine(doc_embs, records, embedder, corpus_texts)
    print(
        f"  SOT Cosine:      recall@5={res_cos['recall@5']:.3f}  recall@1={res_cos['recall@1']:.3f}  MRR={res_cos['mrr']:.3f}  p50={res_cos['latency_p50_ms']:.2f}ms"
    )

    # --- BM25 + SOT Hybrid ---
    res_hybrid = evaluate_bm25_sot_hybrid(records, embedder, corpus_texts, alpha=0.7)
    print(
        f"  BM25+SOT(0.7):   recall@5={res_hybrid['recall@5']:.3f}  recall@1={res_hybrid['recall@1']:.3f}  MRR={res_hybrid['mrr']:.3f}  p50={res_hybrid['latency_p50_ms']:.2f}ms"
    )

    # --- RTMDK + SOT v2 ---
    cfg = RTMDKConfig(
        latent_dim=384,
        top_k=5,
        use_hnsw=False,
        sparse_routing=False,
        adaptive_phase_coupling=True,
    )
    mem = RTMDKMemory(config=cfg, embedder=embedder)

    for rec in records:
        ctx = rec["context"]
        topic = rec.get("topic", "")
        content = {"text": ctx}
        if topic:
            content["topic"] = topic
        mem.add_node(embedder(ctx), content)
    # Also add any contexts that weren't in records
    seen = {r["context"] for r in records}
    for ctx in corpus_texts:
        if ctx not in seen:
            mem.add_node(embedder(ctx), {"text": ctx})

    res_rtmdk = evaluate_rtmdk(mem, records, embedder)
    print(
        f"  RTMDK+SOT:       recall@5={res_rtmdk['recall@5']:.3f}  recall@1={res_rtmdk['recall@1']:.3f}  MRR={res_rtmdk['mrr']:.3f}  p50={res_rtmdk['latency_p50_ms']:.2f}ms"
    )

    return {
        "sot_cosine": res_cos,
        "bm25_sot": res_hybrid,
        "rtmdk_sot": res_rtmdk,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", "en", "ru", "hard", "ms_marco", "rubq"])
    parser.add_argument(
        "--contrastive", action="store_true", help="Contrastive fine-tune SOT v2 on query-context pairs"
    )
    parser.add_argument("--save-model", action="store_true", help="Save trained SOT v2 models to models/")
    parser.add_argument("--max-records", type=int, default=None, help="Limit records per dataset")
    args = parser.parse_args()

    results = {}

    if args.dataset in ("all", "en"):
        records = load_dataset(DATASET_PATHS["qa_1000_en"], language="en", max_records=args.max_records or 200)
        results["qa_1000_en"] = benchmark_dataset("qa_1000_en", records, args.contrastive, save_model=args.save_model)

    if args.dataset in ("all", "ru"):
        records = load_dataset(DATASET_PATHS["comprehensive_500"], language="ru", max_records=args.max_records)
        if records:
            results["comprehensive_500_ru"] = benchmark_dataset(
                "comprehensive_500_ru", records, args.contrastive, save_model=args.save_model
            )

    if args.dataset in ("all", "hard"):
        records = load_dataset(DATASET_PATHS["comprehensive_500"], language="en", max_records=args.max_records)
        results["comprehensive_500"] = benchmark_dataset(
            "comprehensive_500", records, args.contrastive, save_model=args.save_model
        )

    if args.dataset in ("all", "ms_marco"):
        records = load_dataset(DATASET_PATHS["ms_marco"], max_records=args.max_records)
        if records:
            results["ms_marco"] = benchmark_dataset("ms_marco", records, args.contrastive, save_model=args.save_model)

    if args.dataset in ("all", "rubq"):
        records = load_dataset(DATASET_PATHS["rubq"], max_records=args.max_records)
        if records:
            results["rubq"] = benchmark_dataset("rubq", records, args.contrastive, save_model=args.save_model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Fully Self-Contained (SOT v2 only, no SBERT)")
    print("=" * 60)
    for ds_name, res in results.items():
        print(f"\n{ds_name}:")
        for sys_name, metrics in res.items():
            print(
                f"  {sys_name:16s} recall@5={metrics['recall@5']:.3f}  recall@1={metrics['recall@1']:.3f}  MRR={metrics['mrr']:.3f}"
            )


if __name__ == "__main__":
    main()
