"""Benchmark Quantum Resonance Retrieval vs Hybrid BM25+SIF vs SBERT."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.sot_v2.sif_embedder import SIFEmbedder
from rtmdk.memory.sot_v2.hybrid_retriever import HybridSIFBM25Retriever
from rtmdk.memory.sot_v2.quantum import QuantumResonanceRetriever
from scripts.bench_sot_v2 import load_dataset, _build_vocab, _word_tokenize


def bench_all(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    corpus = [r["context"] for r in records] + [r["query"] for r in records]
    vocab = _build_vocab(corpus)
    tok_ctx = [[vocab[w] for w in _word_tokenize(r["context"])] for r in records]
    tok_qry = [[vocab[w] for w in _word_tokenize(r["query"])] for r in records]
    tok_all = tok_ctx + tok_qry

    print("[Train] SIF embedder (a=0.01, window=5)...")
    embedder = SIFEmbedder(latent_dim=384, window_size=5, min_count=1, a=0.01, remove_pc=True)
    embedder.fit(tok_all, len(vocab))

    results = {}

    # --- SBERT baseline ---
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    ctx_embs = np.vstack([model.encode(r["context"], convert_to_numpy=True) for r in records])
    ctx_embs = ctx_embs / (np.linalg.norm(ctx_embs, axis=1, keepdims=True) + 1e-8)
    recalls, ranks, lats = [], [], []
    for i, rec in enumerate(records):
        q = model.encode(rec["query"], convert_to_numpy=True)
        q = q / (np.linalg.norm(q) + 1e-8)
        t0 = time.perf_counter()
        sims = ctx_embs @ q
        top_idx = np.argsort(-sims)[:5]
        lats.append((time.perf_counter() - t0) * 1000)
        recalls.append(1 if i in top_idx else 0)
        rank = np.where(np.argsort(-sims) == i)[0]
        ranks.append(1.0 / (rank[0] + 1) if len(rank) else 0.0)
    results["sbert"] = {
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(ranks)),
        "latency_p50_ms": float(np.median(lats)),
    }

    # --- Hybrid BM25+SIF ---
    ret_hyb = HybridSIFBM25Retriever(latent_dim=384, alpha=0.5)
    for tokens in tok_ctx:
        ret_hyb.add_document(tokens, embedder.embed(tokens))
    recalls, ranks, lats = [], [], []
    for i, rec in enumerate(records):
        q_emb = embedder.embed(tok_qry[i])
        t0 = time.perf_counter()
        res = ret_hyb.query(tok_qry[i], q_emb, top_k=5)
        lats.append((time.perf_counter() - t0) * 1000)
        top_idx = [idx for idx, _ in res]
        recalls.append(1 if i in top_idx else 0)
        rank = None
        for pos, (idx, _) in enumerate(res):
            if idx == i:
                rank = pos
                break
        ranks.append(1.0 / (rank + 1) if rank is not None else 0.0)
    results["hybrid"] = {
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(ranks)),
        "latency_p50_ms": float(np.median(lats)),
    }

    # --- Quantum Resonance ---
    ret_q = QuantumResonanceRetriever(latent_dim=384, epsilon=1e-4, use_coherence=False)
    for i, tokens in enumerate(tok_ctx):
        embs = np.stack([embedder.word_embeddings[t] for t in tokens])
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        embs = embs / norms
        ret_q.add_document(str(i), embs, token_ids=tokens)
    recalls, ranks, lats = [], [], []
    for i, rec in enumerate(records):
        q_tokens = tok_qry[i]
        q_emb = embedder.embed(q_tokens)
        t0 = time.perf_counter()
        res = ret_q.query(q_emb, top_k=5)
        lats.append((time.perf_counter() - t0) * 1000)
        top_idx = [int(did) for did, _ in res]
        recalls.append(1 if i in top_idx else 0)
        rank = None
        for pos, (did, _) in enumerate(res):
            if int(did) == i:
                rank = pos
                break
        ranks.append(1.0 / (rank + 1) if rank is not None else 0.0)
    results["quantum"] = {
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(ranks)),
        "latency_p50_ms": float(np.median(lats)),
    }

    return results


def main():
    records = load_dataset("datasets/qa_1000_en.json", 200)
    print(f"Records: {len(records)}\n")
    results = bench_all(records)
    for name, metrics in results.items():
        print(f"=== {name.upper()} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print()


if __name__ == "__main__":
    main()
