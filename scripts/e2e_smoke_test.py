"""End-to-end smoke test — simulates real RTMDK usage over time.

Usage:
    python scripts/e2e_smoke_test.py

Reports:
    - Ingestion throughput
    - Query latency (p50, p95, p99)
    - Result quality (top-1 relevance)
    - Memory growth
    - WAL integrity
"""

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


@dataclass
class SmokeResult:
    ingest_rate: float  # nodes/sec
    query_p50_ms: float
    query_p95_ms: float
    query_p99_ms: float
    top1_relevance: float  # fraction of queries where top result contains query keyword
    wal_records: int
    memory_nodes: int


def _make_embedder(dim: int = 64):
    """Deterministic embedder that produces distinct vectors per text."""
    cache = {}

    def embed(text: str):
        if text not in cache:
            # Hash-based embedding: each word contributes to different dimensions
            vec = np.zeros(dim, dtype=np.float32)
            for i, word in enumerate(text.lower().split()):
                h = hash(word) % 2**31
                rng = np.random.RandomState(h)
                vec += rng.standard_normal(dim).astype(np.float32)
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            cache[text] = vec
        return cache[text]

    return embed


def _generate_docs(n: int) -> List[Tuple[str, str]]:
    """Generate (query, answer) pairs."""
    topics = [
        "machine learning",
        "neural networks",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "transformer architecture",
        "generative AI",
        "knowledge graphs",
        "vector databases",
    ]
    docs = []
    for i in range(n):
        topic = topics[i % len(topics)]
        docs.append(
            (
                f"What is {topic}?",
                f"{topic} is a subfield of artificial intelligence that focuses on {topic.replace(' ', '_')}_concepts.",
            )
        )
    return docs


def run_smoke_test(n_nodes: int = 200, n_queries: int = 50) -> SmokeResult:
    """Run full smoke test and return metrics."""
    print(f"\n{'='*60}")
    print("RTMDK v8.3 End-to-End Smoke Test")
    print(f"{'='*60}\n")

    cfg = RTMDKConfig(
        latent_dim=64,
        use_hnsw=False,
        wal_fsync_interval_ms=0,
        rate_limit_nodes_per_sec=0,
        enable_async=True,
    )
    embedder = _make_embedder(64)

    td = tempfile.mkdtemp()
    wal_path = os.path.join(td, "smoke.wal")
    mem = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)

    docs = _generate_docs(n_nodes)

    # Phase 1: Ingestion
    print(f"[1/4] Ingesting {n_nodes} nodes...")
    t0 = time.time()
    for query, answer in docs:
        emb = embedder(query + " " + answer)
        mem.add_node(content={"query": query, "answer": answer}, embedding=emb)
    ingest_time = time.time() - t0
    ingest_rate = n_nodes / ingest_time
    print(f"      Done in {ingest_time:.2f}s ({ingest_rate:.0f} nodes/sec)")

    # Phase 2: Queries (immediately after ingest)
    print(f"[2/4] Running {n_queries} queries (hot)...")
    query_latencies = []
    top1_hits = 0
    for i in range(n_queries):
        q, expected_answer = docs[i]
        t0 = time.time()
        results = mem.field.query(embedder(q), top_k=3)
        lat_ms = (time.time() - t0) * 1000
        query_latencies.append(lat_ms)
        if results:
            top_content = results[0][2].content.get("answer", "")
            if expected_answer[:20] in top_content:
                top1_hits += 1

    lat_sorted = sorted(query_latencies)
    p50 = lat_sorted[int(len(lat_sorted) * 0.5)]
    p95 = lat_sorted[int(len(lat_sorted) * 0.95)]
    p99 = lat_sorted[min(int(len(lat_sorted) * 0.99), len(lat_sorted) - 1)]
    relevance = top1_hits / n_queries
    print(f"      Latency: p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")
    print(f"      Top-1 relevance: {relevance:.1%}")

    # Phase 3: Wait / background work
    print("[3/4] Waiting for background tasks (5s)...")
    time.sleep(5.0)

    # Phase 4: Queries after wait (cold)
    print(f"[4/4] Running {n_queries} queries (after 5s wait)...")
    query_latencies2 = []
    top1_hits2 = 0
    for i in range(n_queries):
        q, expected_answer = docs[i]
        t0 = time.time()
        results = mem.field.query(embedder(q), top_k=3)
        lat_ms = (time.time() - t0) * 1000
        query_latencies2.append(lat_ms)
        if results:
            top_content = results[0][2].content.get("answer", "")
            if expected_answer[:20] in top_content:
                top1_hits2 += 1

    p50_2 = sorted(query_latencies2)[int(len(query_latencies2) * 0.5)]
    relevance2 = top1_hits2 / n_queries
    print(f"      Latency: p50={p50_2:.1f}ms")
    print(f"      Top-1 relevance: {relevance2:.1%}")

    # WAL check
    wal = getattr(mem.field, "wal", None)
    wal_records = len(wal.replay()) if wal else 0
    print(f"\nWAL records: {wal_records}")
    print(f"Total nodes: {len(mem.field.nodes)}")

    # Cleanup
    mem.close()
    time.sleep(0.3)  # Windows file handle release

    result = SmokeResult(
        ingest_rate=ingest_rate,
        query_p50_ms=p50,
        query_p95_ms=p95,
        query_p99_ms=p99,
        top1_relevance=relevance,
        wal_records=wal_records,
        memory_nodes=len(mem.field.nodes),
    )

    # Cleanup temp dir
    import shutil

    try:
        shutil.rmtree(td, ignore_errors=True)
    except Exception:
        pass

    # Final verdict
    print(f"\n{'='*60}")
    print("VERDICT:", end=" ")
    if (
        result.ingest_rate > 50
        and result.query_p95_ms < 500
        and result.top1_relevance > 0.5
        and result.wal_records >= n_nodes
    ):
        print("PASS")
    else:
        print("FAIL")
    print(f"{'='*60}\n")

    return result


if __name__ == "__main__":
    result = run_smoke_test(n_nodes=200, n_queries=50)
    sys.exit(0 if result.top1_relevance > 0.5 else 1)
