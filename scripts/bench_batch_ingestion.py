"""
Benchmark: ingest 1M nodes via add_nodes_batch and measure throughput.

Usage:
    python scripts/bench_batch_ingestion.py [--nodes 1000000] [--batch 10000]
"""

import argparse
import time

import numpy as np

from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig


def main():
    parser = argparse.ArgumentParser(description="RTMDK batch ingestion benchmark")
    parser.add_argument("--nodes", type=int, default=1_000_000, help="Total nodes to ingest")
    parser.add_argument("--batch", type=int, default=10_000, help="Batch size per add_nodes_batch call")
    parser.add_argument("--dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--async-hnsw", action="store_true", help="Enable async HNSW build")
    parser.add_argument("--wal-batch", type=int, default=100, help="WAL fsync interval in ms (0 = sync, 100 = async flush)")
    args = parser.parse_args()

    cfg = RTMDKConfig(
        latent_dim=args.dim,
        use_hnsw=True,
        hyperbolic=False,
        bm25_fallback=False,
        quantization="none",
        query_cache_size=0,
        async_hnsw_build=args.async_hnsw,
        async_hnsw_interval_ms=5000,
        async_hnsw_batch_size=10_000,
        wal_fsync_interval_ms=args.wal_batch,
        wal_batch_size=100,
    )
    field = RTMDKField(cfg)

    total = args.nodes
    batch = args.batch
    dim = args.dim

    print(f"Ingesting {total:,} nodes ({dim}d) in batches of {batch:,}")
    print(f"  async_hnsw={args.async_hnsw}, wal_fsync_interval_ms={args.wal_batch}")

    start = time.perf_counter()
    for i in range(0, total, batch):
        n = min(batch, total - i)
        embeddings = np.random.randn(n, dim).astype(np.float32)
        contents = [{"text": f"doc {i + j}"} for j in range(n)]
        field.add_nodes_batch(embeddings, contents)
        if (i + n) % max(batch, 100_000) == 0:
            elapsed = time.perf_counter() - start
            print(f"  {i + n:>10,} nodes | {elapsed:>6.2f}s | {(i + n) / elapsed:>10.1f} nodes/sec")

    elapsed = time.perf_counter() - start
    print(f"\nDone: {total:,} nodes in {elapsed:.2f}s = {total / elapsed:,.1f} nodes/sec")

    if args.async_hnsw:
        print("Flushing async HNSW builder...")
        if field._async_index_builder:
            field._async_index_builder.flush()
        print(f"HNSW positions: {len(field.hnsw_index.positions):,}")

    field.close()


if __name__ == "__main__":
    main()
