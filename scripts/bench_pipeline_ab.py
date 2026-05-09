"""A/B benchmark: pipeline vs legacy retrieval.

Usage:
    python scripts/bench_pipeline_ab.py --queries 100 --top-k 5
    python scripts/bench_pipeline_ab.py --queries 100 --dataset datasets/qa_1000_en.json
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory
from rtmdk.pipeline.ab_testing import PipelineABTester


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


def _load_queries(path: str, limit: int) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        texts = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(item.get("question", item.get("query", item.get("text", ""))))
        return [t for t in texts if t][:limit]
    return []


def main():
    parser = argparse.ArgumentParser(description="A/B benchmark pipeline vs legacy")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", type=str, default=None, help="JSON file with queries")
    parser.add_argument("--nodes", type=int, default=500, help="Number of memory nodes")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup memory
    cfg = RTMDKConfig(
        latent_dim=args.latent_dim,
        embedding_dim=args.latent_dim,
        top_k=args.top_k,
        pipeline_enabled=True,
        pipeline_breaker_enabled=False,
        use_hnsw=False,
    )
    mem = RTMDKMemory(config=cfg, embedder=_make_embedder(args.latent_dim))

    print(f"Building memory field with {args.nodes} nodes...")
    for i in range(args.nodes):
        text = f"document about topic {i % 50} aspect {i}"
        emb = _make_embedder(args.latent_dim)(text)
        mem.add_node(embedding=emb, content={"text": text}, node_id=f"n{i}")

    # Load or generate queries
    if args.dataset:
        queries = _load_queries(args.dataset, args.queries)
    else:
        topics = [f"topic {i % 50}" for i in range(args.queries)]
        queries = [f"document about {t}" for t in topics]

    print(f"Running A/B test on {len(queries)} queries (top_k={args.top_k})...")

    tester = PipelineABTester(mem)
    results = tester.compare_batch(queries, top_k=args.top_k)
    summary = tester.summary()

    print("\n" + "=" * 60)
    print("A/B TEST RESULTS")
    print("=" * 60)
    print(f"Queries: {summary['runs']}")
    print(f"\nLatency (ms):")
    print(f"  Legacy:   mean={summary['legacy_latency_ms']['mean']:.2f}  "
          f"median={summary['legacy_latency_ms']['median']:.2f}  "
          f"p95={summary['legacy_latency_ms']['p95']:.2f}")
    print(f"  Pipeline: mean={summary['pipeline_latency_ms']['mean']:.2f}  "
          f"median={summary['pipeline_latency_ms']['median']:.2f}  "
          f"p95={summary['pipeline_latency_ms']['p95']:.2f}")
    print(f"\nPipeline faster: {summary['pipeline_faster_count']} / {summary['runs']} "
          f"({100*summary['pipeline_faster_count']/max(summary['runs'],1):.1f}%)")
    print(f"\nResult quality:")
    print(f"  Jaccard overlap: mean={summary['jaccard_overlap']['mean']:.4f}  "
          f"median={summary['jaccard_overlap']['median']:.4f}")
    print(f"  Kendall tau:     mean={summary['kendall_tau']['mean']:.4f}  "
          f"median={summary['kendall_tau']['median']:.4f}")

    # Per-query details
    print("\n" + "-" * 60)
    print("Per-query samples (first 5):")
    for r in results[:5]:
        comp = r["comparison"]
        print(f"  '{r['query'][:40]}...'  "
              f"jaccard={comp['jaccard_overlap']:.2f}  "
              f"tau={comp['kendall_tau']:.2f}  "
              f"legacy={r['legacy']['latency_ms']:.2f}ms  "
              f"pipeline={r['pipeline']['latency_ms']:.2f}ms  "
              f"faster={comp['faster']}")

    print("\n" + "=" * 60)
    if summary['pipeline_faster_count'] >= summary['runs'] * 0.8:
        print("[PASS] Pipeline is consistently faster -- ready for production")
    elif summary['jaccard_overlap']['mean'] >= 0.95:
        print("[PASS] Results are nearly identical -- pipeline is safe to enable")
    else:
        print("[WARN] Significant differences detected -- review before enabling")


if __name__ == "__main__":
    main()
