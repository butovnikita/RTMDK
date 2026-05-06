"""
Quantization research: test fp16 recall vs float32 baseline.

Uses built-in RTMDK quantization (quantization="fp16") and manual
int8 quantization for comparison.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
np.random.seed(42)

DIM = 256
N_NODES = 10_000
N_QUERIES = 500
TOP_K = 5


def build_field(positions, quantize="none"):
    cfg = RTMDKConfig(
        latent_dim=DIM, top_k=TOP_K, min_response=0.001,
        decay_rate=0.999, use_hnsw=False, learn_projection=False,
        bm25_fallback=False, enable_async=False,
        resonance_kernel="cosine", phase_coupling=0.0,
        quantization=quantize,
    )
    field = RTMDKField(cfg)

    # Manual int8 simulation (not yet built-in)
    if quantize == "int8_global":
        positions = _manual_int8(positions, mode="global")
    elif quantize == "int8_per_dim":
        positions = _manual_int8(positions, mode="per_dim")

    for i in range(N_NODES):
        field.add_node(positions[i], content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
        field.nodes[field.node_index[-1]].amplitude = 1.0
        field.nodes[field.node_index[-1]].salience = 1.0
    return field


def _manual_int8(positions, mode="global"):
    if mode == "global":
        scale = 1.0 / 127.0
        return np.round(positions / scale).clip(-127, 127).astype(np.int8).astype(np.float32) * scale
    else:
        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        scales = (maxs - mins) / 255.0
        scales = np.maximum(scales, 1e-8)
        return np.round((positions - mins) / scales).clip(0, 255).astype(np.uint8).astype(np.float32) * scales + mins


def query_field(field, queries, top_k=5):
    results = []
    for q in queries:
        r = field.query(q, top_k=top_k)
        results.append([nid for nid, _, _ in r])
    return results


def compute_recall(approx, exact, k=1):
    hits = 0
    total = 0
    for a, e in zip(approx, exact):
        a_set = set(a[:k])
        e_set = set(e[:k])
        hits += len(a_set & e_set)
        total += len(e_set)
    return hits / total if total > 0 else 0.0


def estimate_memory_mb(field):
    """Rough estimate of embedding memory (nodes + cache)."""
    n = len(field.node_index)
    d = field.cfg.latent_dim
    itemsize = field._quant.itemsize
    # Nodes embeddings + cached_positions (both same dtype)
    mb = (n * d * itemsize * 2) / (1024 * 1024)
    return mb


def main():
    print("Quantization Research: 10K nodes, 256d, 500 queries")
    print("=" * 60)

    positions = np.random.randn(N_NODES, DIM).astype(np.float32)
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)

    queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # Baseline
    print("\n[Baseline] float32")
    field_base = build_field(positions, "none")
    exact_results = query_field(field_base, queries, top_k=TOP_K)
    base_mem = estimate_memory_mb(field_base)
    print(f"  Baseline computed  (~{base_mem:.1f} MB embeddings)")

    results = {"baseline": {"r1": 1.0, "r5": 1.0, "mem": base_mem}}

    for name in ["fp16", "int8_global", "int8_per_dim"]:
        print(f"\n[{name}]")
        field_q = build_field(positions, name)
        approx = query_field(field_q, queries, top_k=TOP_K)
        r1 = compute_recall(approx, exact_results, k=1)
        r5 = compute_recall(approx, exact_results, k=5)
        mem = estimate_memory_mb(field_q)
        print(f"  Recall: R@1={r1:.4f} R@5={r5:.4f}  (~{mem:.1f} MB embeddings)")
        results[name] = {"r1": r1, "r5": r5, "mem": mem}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        reduction = r["mem"] / base_mem if base_mem > 0 else 1.0
        print(f"  {name:15s}: R@1={r['r1']:.4f} R@5={r['r5']:.4f}  mem={r['mem']:.1f}MB ({reduction:.2f}x)")


if __name__ == "__main__":
    main()
