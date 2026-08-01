#!/usr/bin/env python3
"""Grid search RTMDK hyperparameters on BEIR datasets."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bench_beir_rtmdk import (
    load_beir_dataset,
    embed_corpus_and_queries,
    build_rtmdk,
    evaluate_rtmdk,
    evaluate_cosine,
)


def grid_search(dataset: str, model: str, batch_size: int):
    corpus, queries, qrels = load_beir_dataset(dataset)
    corpus_ids, corpus_embs, query_ids, query_embs = embed_corpus_and_queries(corpus, queries, model, batch_size)

    # Baseline
    cosine = evaluate_cosine(corpus_embs, query_embs, query_ids, qrels, corpus_ids)
    print(f"Cosine baseline: recall@1={cosine['recall@1']:.4f} mrr={cosine['mrr']:.4f}")

    best = {"mrr": 0.0, "cfg": None, "metrics": None}
    configs = []
    for bw in [0.5, 0.8, 1.0, 1.2, 1.5]:
        for pc in [0.0, 0.1, 0.2, 0.3, 0.5]:
            for hyp in [False, True]:
                configs.append({"bandwidth": bw, "phase_coupling": pc, "hyperbolic": hyp})

    for cfg in configs:
        mem = build_rtmdk(corpus_ids, corpus_embs, **cfg)
        metrics = evaluate_rtmdk(mem, query_ids, query_embs, qrels, corpus_ids)
        print(
            f"  bw={cfg['bandwidth']:.1f} pc={cfg['phase_coupling']:.1f} hyp={cfg['hyperbolic']:<5} "
            f"recall@1={metrics['recall@1']:.4f} mrr={metrics['mrr']:.4f}"
        )
        if metrics["mrr"] > best["mrr"]:
            best = {"mrr": metrics["mrr"], "cfg": cfg, "metrics": metrics}

    print(f"\nBest config: {best['cfg']}")
    print(f"Best MRR: {best['mrr']:.4f} (cosine={cosine['mrr']:.4f})")
    print(f"Best recall@1: {best['metrics']['recall@1']:.4f} (cosine={cosine['recall@1']:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    grid_search(args.dataset, args.model, args.batch_size)
