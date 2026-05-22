"""Benchmark quantization modes: recall vs memory trade-off.

Usage:
    python scripts/bench_quantization_recall.py --dataset datasets/qa_1000_en.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


def load_qa_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_field(dataset, quant_mode: str, dim: int = 64):
    cfg = RTMDKConfig(
        latent_dim=dim,
        embedding_dim=dim,
        quantization=quant_mode,
        use_hnsw=False,
    )
    field = RTMDKField(config=cfg)
    embedder = _make_embedder(dim)
    for item in dataset:
        emb = embedder(item["query"])
        field.add_node(emb, {"text": item["answer"]})
    return field, embedder


def evaluate_recall(field, dataset, embedder, k: int = 1):
    correct = 0
    latencies = []
    for item in dataset:
        emb = embedder(item["query"])
        t0 = time.perf_counter()
        results = field.query(emb, top_k=k)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if results and any(item["answer"] in (r.get("content", {}).get("text", "") for r in results) for r in results):
            correct += 1
    recall = correct / len(dataset) if dataset else 0.0
    return recall, np.mean(latencies)


def measure_ram(field):
    import gc
    import os
    try:
        import psutil
        gc.collect()
        process = psutil.Process(os.getpid())
        return process.memory_info().rss // (1024 * 1024)
    except ImportError:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/qa_1000_en.json")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()

    dataset = load_qa_dataset(args.dataset)
    # Use subset for speed
    dataset = dataset[:500]

    modes = ["none", "fp16", "int8", "int8_global", "int8_per_dim"]
    print(f"{'Mode':<15} {'Recall@1':>10} {'Latency':>10} {'RAM(MB)':>10}")
    print("-" * 50)
    for mode in modes:
        field, embedder = build_field(dataset, mode, args.dim)
        recall, latency = evaluate_recall(field, dataset, embedder, args.k)
        ram = measure_ram(field)
        print(f"{mode:<15} {recall:>10.3f} {latency:>10.2f} {ram:>10}")
        field.nodes.close()


if __name__ == "__main__":
    main()
