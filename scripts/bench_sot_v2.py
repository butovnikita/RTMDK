"""Benchmark SOT v2.0: Word-level tokenization + SIF sentence embeddings.

No external embedder required.  Theoretical foundation:
    - Arora et al. 2017: SIF = Smooth Inverse Frequency + PCA removal
    - Levy & Goldberg 2014: PMI matrix factorisation = word embeddings

Measures Recall@K and MRR on QA dataset.
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

from rtmdk.memory.sot_v2.sif_embedder import SIFEmbedder
from rtmdk.memory.sot_v2.hybrid_retriever import HybridSIFBM25Retriever


def load_dataset(path: str, n: int) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data
    return records[:n]


def _word_tokenize(text: str) -> List[str]:
    """Simple unicode-aware word tokenization."""
    import unicodedata

    text = text.lower()
    tokens = []
    current = []
    for ch in text:
        is_cjk = (
            "\u4e00" <= ch <= "\u9fff"
            or "\u3040" <= ch <= "\u309f"
            or "\u30a0" <= ch <= "\u30ff"
            or "\uac00" <= ch <= "\ud7af"
        )
        if is_cjk:
            if current:
                tokens.append("".join(current))
                current = []
            tokens.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _build_vocab(texts: List[str]) -> Dict[str, int]:
    vocab = {}
    for text in texts:
        for word in _word_tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def bench_sot_v2(records: List[Dict], top_k: int = 5) -> Dict[str, float]:
    print("[SOT v2.0] Building word vocabulary...")
    corpus = [r["context"] for r in records] + [r["query"] for r in records]
    vocab = _build_vocab(corpus)
    print(f"[SOT v2.0] Vocab size: {len(vocab)}")

    print("[SOT v2.0] Tokenizing corpus...")
    tokenized_contexts = [[vocab[w] for w in _word_tokenize(r["context"])] for r in records]
    tokenized_queries = [[vocab[w] for w in _word_tokenize(r["query"])] for r in records]

    print("[SOT v2.0] Fitting SIF embedder (PMI + SVD + SIF weights)...")
    embedder = SIFEmbedder(latent_dim=384, window_size=5, min_count=1, a=1e-3, remove_pc=True)
    all_tokenized = tokenized_contexts + tokenized_queries
    embedder.fit(all_tokenized, vocab_size=len(vocab))

    print("[SOT v2.0] Building hybrid BM25 + SIF index...")
    retriever = HybridSIFBM25Retriever(latent_dim=384, alpha=0.5)
    for i, tokens in enumerate(tokenized_contexts):
        emb = embedder.embed(tokens)
        retriever.add_document(tokens, emb)

    print(f"[SOT v2.0] Retrieving top-{top_k} for {len(records)} queries...")
    recalls = []
    ranks = []
    latencies = []
    for i, rec in enumerate(records):
        q_emb = embedder.embed(tokenized_queries[i])
        t0 = time.perf_counter()
        results = retriever.query(tokenized_queries[i], q_emb, top_k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)
        top_idx = [idx for idx, _ in results]
        recalls.append(1 if i in top_idx else 0)
        rank = None
        for pos, (idx, _) in enumerate(results):
            if idx == i:
                rank = pos
                break
        ranks.append(1.0 / (rank + 1) if rank is not None else 0.0)

    return {
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(ranks)),
        "latency_p50_ms": float(np.median(latencies)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }


def bench_sbert(records: List[Dict], top_k: int = 5) -> Dict[str, float]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    contexts = [r["context"] for r in records]
    queries = [r["query"] for r in records]
    ctx_embs = np.vstack([model.encode(c, convert_to_numpy=True) for c in contexts])
    ctx_embs = ctx_embs / (np.linalg.norm(ctx_embs, axis=1, keepdims=True) + 1e-8)

    recalls = []
    ranks = []
    latencies = []
    for i, query in enumerate(queries):
        q_emb = model.encode(query, convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/qa_1000_en.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    records = load_dataset(args.dataset, args.n)
    print(f"Records: {len(records)}\n")

    print("=== SBERT Baseline ===")
    sbert = bench_sbert(records, top_k=args.top_k)
    for k, v in sbert.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== SOT v2.0 (Word-level + PMI + SIF) ===")
    sot = bench_sot_v2(records, top_k=args.top_k)
    for k, v in sot.items():
        print(f"  {k}: {v:.4f}")
    print(
        f"  vs SBERT — recall delta={sot['recall_at_k']-sbert['recall_at_k']:+.3f}, mrr delta={sot['mrr']-sbert['mrr']:+.3f}"
    )


if __name__ == "__main__":
    main()
