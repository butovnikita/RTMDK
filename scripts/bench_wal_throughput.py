"""Benchmark WAL throughput: sync (fsync_interval_ms=0) vs async (100ms)."""

import os
import sys
import tempfile
import time

import numpy as np

os.environ["RTMDK_ADD_RATE_LIMIT"] = "100000"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.core import RTMDKMemory  # noqa: E402
from rtmdk.memory.config import RTMDKConfig  # noqa: E402


def _embedder(text: str):
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(64).astype(np.float32)


def benchmark_wal(fsync_ms: int, n_nodes: int = 1000):
    td = tempfile.mkdtemp()
    wal_path = os.path.join(td, "wal.jsonl")
    cfg = RTMDKConfig(
        latent_dim=64,
        use_hnsw=False,
        wal_fsync_interval_ms=fsync_ms,
        wal_batch_size=100,
    )
    mem = RTMDKMemory(config=cfg, embedder=_embedder, wal_path=wal_path)

    start = time.perf_counter()
    for i in range(n_nodes):
        emb = np.random.randn(64).astype(np.float32)
        mem.add_node(
            content={"text": "node {}".format(i), "topic": "test"},
            embedding=emb,
        )
    elapsed = time.perf_counter() - start

    # Verify WAL replay works
    mem2 = RTMDKMemory(config=cfg, embedder=_embedder, wal_path=wal_path)
    replay_count = len(mem2.field.nodes)

    # Explicit cleanup to avoid Windows file-lock issues
    mem.field.wal.close()
    mem2.field.wal.close()
    try:
        os.remove(wal_path)
        os.rmdir(td)
    except OSError:
        pass

    return elapsed, replay_count


def main():
    print("=" * 60)
    print("WAL Throughput Benchmark")
    print("=" * 60)

    n_nodes = 2000

    print("\nNodes: {}".format(n_nodes))
    print("-" * 40)

    # Sync WAL
    elapsed_sync, replay_sync = benchmark_wal(0, n_nodes)
    throughput_sync = n_nodes / elapsed_sync
    print("Sync WAL  (fsync every write):")
    print("  Time:       {:.2f}s".format(elapsed_sync))
    print("  Throughput: {:,.0f} nodes/sec".format(throughput_sync))
    print("  Replay OK:  {}".format(replay_sync == n_nodes))

    # Async WAL 100ms
    elapsed_async, replay_async = benchmark_wal(100, n_nodes)
    throughput_async = n_nodes / elapsed_async
    print("\nAsync WAL (fsync every 100ms):")
    print("  Time:       {:.2f}s".format(elapsed_async))
    print("  Throughput: {:,.0f} nodes/sec".format(throughput_async))
    print("  Replay OK:  {}".format(replay_async == n_nodes))

    speedup = throughput_async / throughput_sync
    print("\nSpeedup: {:.1f}x".format(speedup))
    print("=" * 60)


if __name__ == "__main__":
    main()
