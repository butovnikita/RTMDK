"""
Comprehensive grid search for adaptive bandwidth parameters.

Optimizes by building the field ONCE, then only recomputing _cached_bw
for each (transform, clip, k) combination.

Metrics:
- R@1, R@5
- BW spread (p99/p01 ratio)
- Median absolute deviation of BW
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
from sentence_transformers import SentenceTransformer
from scipy.spatial import cKDTree
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
np.random.seed(42)


def load_dataset(path="datasets/qa_1000_en.json", n=300):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)["records"]
    return data[:n]


def build_field(records, cfg, model):
    field = RTMDKField(cfg)
    texts = [r["query"] + " " + r["answer"] for r in records]
    print(f"  Encoding {len(texts)} texts...")
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for rec, emb in zip(records, embs):
        field.add_node(
            emb.astype(np.float32),
            content={"text": rec["answer"]},
            phase=0.0,
            node_id=f"n{hash(rec['query'] + rec['answer']) & 0x7FFFFFFF}",
            skip_projection=True,
        )
        field.nodes[field.node_index[-1]].amplitude = 1.0
        field.nodes[field.node_index[-1]].salience = 1.0
    return field


def evaluate(field, records, model, top_k=5):
    correct_1 = 0
    correct_k = 0
    total = 0
    for rec in records:
        q_emb = model.encode(
            rec["query"],
            convert_to_numpy=True).astype(
            np.float32)
        results = field.query(q_emb, top_k=top_k)
        if not results:
            continue
        top_text = results[0][2].content.get("text", "")
        if top_text == rec["answer"]:
            correct_1 += 1
        found = any(r[2].content.get("text") == rec["answer"] for r in results)
        if found:
            correct_k += 1
        total += 1
    return {"R@1": correct_1 / total, "R@5": correct_k / total}


def compute_bw(positions, global_bw, k, transform, clip_range):
    """Compute adaptive bw vector. positions: (N, D) array."""
    n = len(positions)
    if n <= k:
        return None
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=k + 1)
    kdist = distances[:, -1].astype(np.float32)
    median_kdist = float(np.median(kdist))

    if transform == "sqrt":
        factors = np.sqrt(kdist / max(median_kdist, 1e-8))
    elif transform == "linear":
        factors = kdist / max(median_kdist, 1e-8)
    elif transform == "log":
        factors = np.log1p(kdist / max(median_kdist, 1e-8))
    elif transform == "power1_3":
        factors = (kdist / max(median_kdist, 1e-8)) ** (1.0 / 3.0)
    elif transform == "power2_3":
        factors = (kdist / max(median_kdist, 1e-8)) ** (2.0 / 3.0)
    elif transform == "percentile":
        # Use percentile rank instead of raw ratio
        ranks = np.argsort(np.argsort(kdist))
        factors = 0.1 + 1.9 * (ranks / max(len(ranks) - 1, 1))
    else:
        raise ValueError(f"Unknown transform: {transform}")

    if clip_range is not None:
        lo, hi = clip_range
        factors = np.clip(factors, lo, hi)

    return (global_bw * factors).astype(np.float32)


def run_grid_search(field, records, model, base_stats):
    """Grid search over k, transform, clip_range."""

    # k values
    ks = [5, 10, 15, 20, 30, 50]

    # transforms
    transforms = [
        "sqrt",
        "linear",
        "log",
        "power1_3",
        "power2_3",
        "percentile"]

    # clip ranges: (lo, hi) or None
    clips = [
        None,
        (0.5, 2.0),
        (0.2, 5.0),
        (0.1, 10.0),
        (0.05, 20.0),
        (0.01, 100.0),
        (0.001, 1000.0),
    ]

    total = len(ks) * len(transforms) * len(clips)
    print(f"\nGrid search: {total} configs")
    print("=" * 80)
    print(f"{'k':>3} {'transform':>12} {'clip':>16} {'R@1':>6} {'R@5':>6} {'spread':>8} {'delta':>8}")
    print("=" * 80)

    positions = field._cached_positions.copy()
    global_bw = field.cfg.bandwidth

    best = {"R@1": 0, "config": None}

    for k in ks:
        for transform in transforms:
            for clip_range in clips:
                bw = compute_bw(positions, global_bw, k, transform, clip_range)
                if bw is None:
                    continue

                field._cached_bw = bw
                stats = evaluate(field, records, model, top_k=5)
                spread = float(np.percentile(bw, 99) /
                               max(np.percentile(bw, 1), 1e-8))
                delta_r1 = stats["R@1"] - base_stats["R@1"]

                clip_str = f"{clip_range[0]}-{clip_range[1]}" if clip_range else "none"
                print(
                    f"{k:>3} {transform:>12} {clip_str:>16} {stats['R@1']:>6.3f} {stats['R@5']:>6.3f} {spread:>8.1f}x {delta_r1:>+7.3f}")

                if stats["R@1"] > best["R@1"]:
                    best = {
                        "R@1": stats["R@1"],
                        "config": (k, transform, clip_range),
                        "spread": spread,
                        "stats": stats,
                    }

    print("=" * 80)
    print(f"\nBEST CONFIG:")
    k, transform, clip_range = best["config"]
    clip_str = f"{clip_range[0]}-{clip_range[1]}" if clip_range else "none"
    print(f"  k={k}, transform={transform}, clip={clip_str}")
    print(
        f"  R@1={best['stats']['R@1']:.3f} (baseline={base_stats['R@1']:.3f})")
    print(f"  spread={best['spread']:.1f}x")

    return best


def main():
    records = load_dataset(n=300)
    print(f"Dataset: {len(records)} QA records")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Baseline: global bandwidth
    print("\n" + "=" * 80)
    print("Building baseline field (global bw)...")
    cfg = RTMDKConfig(
        latent_dim=384,
        bandwidth=1.0,
        adaptive_bandwidth=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    field = build_field(records, cfg, model)
    field._build_node_cache()
    base_stats = evaluate(field, records, model, top_k=5)
    print(
        f"Baseline R@1: {base_stats['R@1']:.3f}  R@5: {base_stats['R@5']:.3f}")

    # Enable adaptive flag so field uses _cached_bw
    field.cfg.adaptive_bandwidth = True

    # Grid search
    best = run_grid_search(field, records, model, base_stats)

    # Also test with projection 384->128
    print("\n" + "=" * 80)
    print("Testing with projection 384->128 (production config)...")
    cfg_proj = RTMDKConfig(
        latent_dim=128,
        bandwidth=1.0,
        adaptive_bandwidth=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
        learn_projection=False,
    )
    field_proj = build_field(records, cfg_proj, model)
    field_proj._build_node_cache()
    base_proj = evaluate(field_proj, records, model, top_k=5)
    print(f"Projection baseline R@1: {base_proj['R@1']:.3f}")

    field_proj.cfg.adaptive_bandwidth = True
    k, transform, clip_range = best["config"]
    bw_proj = compute_bw(field_proj._cached_positions,
                         1.0, k, transform, clip_range)
    if bw_proj is not None:
        field_proj._cached_bw = bw_proj
        proj_stats = evaluate(field_proj, records, model, top_k=5)
        print(
            f"Projection adaptive  R@1: {proj_stats['R@1']:.3f}  delta={proj_stats['R@1'] - base_proj['R@1']:+.3f}")


if __name__ == "__main__":
    main()
