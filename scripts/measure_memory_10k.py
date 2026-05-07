"""
Measure memory footprint and token economics at 10K nodes.

Metrics:
1. RAM per node (field + HNSW + metadata)
2. Total RTMDK memory at 10K nodes
3. Token savings vs naive context stuffing
4. Comparison: exact vs HNSW vs SOT-only
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
np.random.seed(42)

DIM = 256
N_NODES = 10_000


def get_ram_mb():
    import psutil
    return psutil.Process().memory_info().rss / (1024 * 1024)


def build_field(cfg_name, cfg):
    import gc
    print(f"\n{'='*60}")
    print(f"Config: {cfg_name}")
    print(f"{'='*60}")

    gc.collect()
    ram_before = get_ram_mb()
    field = RTMDKField(cfg)

    # Generate random normalized embeddings + realistic text content
    positions = np.random.randn(N_NODES, DIM).astype(np.float32)
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)

    t0 = time.time()
    for i in range(N_NODES):
        text = f"This is sample document number {i} with some content about topic {i % 100}"
        field.add_node(
            positions[i],
            content={"text": text, "id": i},
            phase=0.0,
            node_id=f"n{i}",
            skip_projection=True,
        )
        field.nodes[field.node_index[-1]].amplitude = 1.0
        field.nodes[field.node_index[-1]].salience = 1.0
    build_time = time.time() - t0

    ram_after = get_ram_mb()
    ram_total = ram_after - ram_before
    ram_per_node = ram_total * 1024 / N_NODES  # KB

    # Query latency
    queries = np.random.randn(100, DIM).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        field.query(q, top_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)
    latencies = np.array(latencies)

    # Token economics
    avg_chars_per_node = sum(len(field.nodes[f"n{i}"].content.get(
        "text", "")) for i in range(N_NODES)) / N_NODES
    avg_tokens_per_node = avg_chars_per_node / 4  # ~4 chars per token (rough)
    tokens_naive = N_NODES * avg_tokens_per_node
    tokens_rtmdk = 5 * avg_tokens_per_node
    savings_ratio = tokens_naive / tokens_rtmdk

    print(f"  Build time: {build_time:.1f}s")
    print(f"  RAM total: {ram_total:.1f} MB")
    print(f"  RAM per node: {ram_per_node:.2f} KB")
    print(
        f"  Query latency: p50={np.percentile(latencies, 50):.2f}ms p99={np.percentile(latencies, 99):.2f}ms")
    print(f"  Avg tokens/node: {avg_tokens_per_node:.0f}")
    print(
        f"  Token savings vs naive: {savings_ratio:.0f}x (naive={tokens_naive:.0f} -> RTMDK={tokens_rtmdk:.0f})")

    del field
    import gc
    gc.collect()

    return {
        "ram_total_mb": ram_total,
        "ram_per_node_kb": ram_per_node,
        "query_p50_ms": float(np.percentile(latencies, 50)),
        "query_p99_ms": float(np.percentile(latencies, 99)),
        "token_savings_ratio": savings_ratio,
    }


def main():
    try:
        pass
    except ImportError:
        print("pip install psutil required for memory measurement")
        return

    print(f"RTMDK Memory & Token Economics Benchmark")
    print(f"Nodes: {N_NODES:,} | Dim: {DIM}")

    results = {}

    # 1. Exact (no HNSW, no SOT)
    results["exact"] = build_field("Exact (no HNSW)", RTMDKConfig(
        latent_dim=DIM, top_k=5, min_response=0.001,
        decay_rate=0.999, use_hnsw=False, learn_projection=False,
        bm25_fallback=False, enable_async=False,
        resonance_kernel="cosine", phase_coupling=0.0,
    ))

    # 2. HNSW
    results["hnsw"] = build_field("HNSW", RTMDKConfig(
        latent_dim=DIM, top_k=5, min_response=0.001,
        decay_rate=0.999, use_hnsw=True, learn_projection=False,
        bm25_fallback=False, enable_async=False,
        resonance_kernel="cosine", phase_coupling=0.0,
        hnsw_m=32, hnsw_ef_construction=400,
    ))

    # 3. SOT (word-level, no external embedder)
    results["sot"] = build_field("SOT word-level", RTMDKConfig(
        latent_dim=DIM, top_k=5, min_response=0.001,
        decay_rate=0.999, use_hnsw=False, learn_projection=False,
        bm25_fallback=False, enable_async=False,
        resonance_kernel="cosine", phase_coupling=0.0,
        sot_enabled=True, sot_use_for_query=True,
        sot_tokenization_mode="word",
        sot_max_vocab=5000,
    ))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        print(
            f"  {name:15s}: RAM={r['ram_total_mb']:5.1f}MB  per_node={r['ram_per_node_kb']:5.2f}KB  "
            f"p50={r['query_p50_ms']:5.2f}ms  savings={r['token_savings_ratio']:4.0f}x")

    # Write JSON
    with open("archive/benchmarks/memory_10k_report.json", "w", encoding="utf-8") as f:
        json.dump({"N_NODES": N_NODES, "DIM": DIM, "results": results},
                  f, indent=2, ensure_ascii=False)
    print("\nReport saved to archive/benchmarks/memory_10k_report.json")


if __name__ == "__main__":
    main()
