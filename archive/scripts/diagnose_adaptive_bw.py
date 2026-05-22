"""
Diagnostic script for adaptive_bandwidth accuracy degradation.

Hypotheses:
1. Clip ranges too aggressive / too loose
2. sqrt() transform over/under-corrects for density
3. k too small for reliable local density estimation in high-D
4. Random projection distorts local density structure
5. Median normalization biased by outliers
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
from scipy import stats
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


np.random.seed(42)


def generate_clustered_embeddings(n_clusters=10, pts_per_cluster=50, dim=128, outlier_ratio=0.1):
    """Generate embeddings where each cluster has a known center."""
    embeddings = []
    labels = []
    centers = []
    rng = np.random.default_rng(42)
    for i in range(n_clusters):
        center = rng.standard_normal(dim).astype(np.float32)
        center /= np.linalg.norm(center)
        centers.append(center)
        # Cluster points: Gaussian around center, then normalize
        pts = rng.standard_normal((pts_per_cluster, dim)).astype(np.float32) * 0.05
        pts += center
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        embeddings.append(pts)
        labels.extend([i] * pts_per_cluster)
    # Outliers: random uniform on sphere
    n_outliers = int(n_clusters * pts_per_cluster * outlier_ratio)
    outliers = rng.standard_normal((n_outliers, dim)).astype(np.float32)
    outliers = outliers / np.linalg.norm(outliers, axis=1, keepdims=True)
    embeddings.append(outliers)
    labels.extend([-1] * n_outliers)
    X = np.vstack(embeddings)
    return X, np.array(labels), np.array(centers)


def build_field(X, labels, cfg):
    field = RTMDKField(cfg)
    for i, pos in enumerate(X):
        field.add_node(
            pos.astype(np.float32),
            content={"cluster": int(labels[i])},
            phase=0.0,
            node_id=f"n{i}",
            skip_projection=True,
        )
        field.nodes[f"n{i}"].amplitude = 1.0
        field.nodes[f"n{i}"].salience = 1.0
    return field


def evaluate(field, centers, labels, top_k=5):
    """Query from each cluster center. Expect top result from same cluster."""
    correct_at_1 = 0
    correct_at_k = 0
    total = 0
    for ci, center in enumerate(centers):
        results = field.query(center.astype(np.float32), top_k=top_k)
        if not results:
            continue
        top_ids = [nid for nid, _, _ in results]
        top_labels = [labels[int(nid[1:])] for nid in top_ids]  # "n123" -> 123
        if top_labels[0] == ci:
            correct_at_1 += 1
        if ci in top_labels:
            correct_at_k += 1
        total += 1
    return {
        "R@1": correct_at_1 / total if total else 0,
        "R@5": correct_at_k / total if total else 0,
        "n": total,
    }


def bw_stats(field):
    if field._cached_bw is None:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    bw = field._cached_bw
    return {
        "mean": float(np.mean(bw)),
        "median": float(np.median(bw)),
        "min": float(np.min(bw)),
        "max": float(np.max(bw)),
        "std": float(np.std(bw)),
        "p10": float(np.percentile(bw, 10)),
        "p90": float(np.percentile(bw, 90)),
    }


def run_experiment(name, cfg, X, labels, centers):
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")
    field = build_field(X, labels, cfg)
    # force cache build
    field._build_node_cache()
    stats = evaluate(field, centers, labels, top_k=5)
    bw = bw_stats(field)
    print(f"  R@1: {stats['R@1']:.3f}  R@5: {stats['R@5']:.3f}")
    if bw["mean"] is not None:
        print(f"  BW  mean={bw['mean']:.4f} med={bw['median']:.4f} std={bw['std']:.4f}")
        print(f"  BW  min={bw['min']:.4f} max={bw['max']:.4f} p10={bw['p10']:.4f} p90={bw['p90']:.4f}")
    else:
        print("  BW  (not computed — adaptive disabled or n too small)")
    return field, stats, bw


def main():
    print("Generating synthetic dataset...")
    X, labels, centers = generate_clustered_embeddings(n_clusters=10, pts_per_cluster=50, dim=128, outlier_ratio=0.1)
    print(f"  Total nodes: {len(X)}  Clusters: 10  Outliers: {sum(labels==-1)}")

    # Baseline: global bandwidth
    cfg_base = RTMDKConfig(
        latent_dim=128,
        bandwidth=1.0,
        adaptive_bandwidth=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f_base, s_base, bw_base = run_experiment("Baseline (global bw=1.0)", cfg_base, X, labels, centers)

    # Current adaptive: k=5, default clip
    cfg_adapt = RTMDKConfig(
        latent_dim=128,
        bandwidth=1.0,
        adaptive_bandwidth=True,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f_adapt, s_adapt, bw_adapt = run_experiment("Adaptive (k=5, current clip)", cfg_adapt, X, labels, centers)

    # Test: larger k
    cfg_adapt_k10 = RTMDKConfig(
        latent_dim=128,
        bandwidth=1.0,
        adaptive_bandwidth=True,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f_adapt_k10, s_adapt_k10, bw_adapt_k10 = run_experiment("Adaptive (k=10)", cfg_adapt_k10, X, labels, centers)

    # Test: larger k=20
    cfg_adapt_k20 = RTMDKConfig(
        latent_dim=128,
        bandwidth=1.0,
        adaptive_bandwidth=True,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f_adapt_k20, s_adapt_k20, bw_adapt_k20 = run_experiment("Adaptive (k=20)", cfg_adapt_k20, X, labels, centers)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rows = [
        ("Baseline (global)", s_base),
        ("Adaptive k=5", s_adapt),
        ("Adaptive k=10", s_adapt_k10),
        ("Adaptive k=20", s_adapt_k20),
    ]
    for name, st in rows:
        print(f"  {name:25s}  R@1={st['R@1']:.3f}  R@5={st['R@5']:.3f}")

    # Check if adaptive bw factors correlate with anything meaningful
    if f_adapt._cached_bw is not None:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC: BW vs distance to cluster center")
        print("=" * 60)
        # For each node, compute distance to its cluster center
        dists_to_center = []
        bw_values = []
        for i, pos in enumerate(X):
            cl = labels[i]
            if cl >= 0:
                d = float(np.linalg.norm(pos - centers[cl]))
                dists_to_center.append(d)
                idx = f_adapt.node_index.index(f"n{i}")
                bw_values.append(float(f_adapt._cached_bw[idx]))
        corr, pval = stats.pearsonr(dists_to_center, bw_values)
        print(f"  Pearson r(dist_to_center, bw) = {corr:.3f} (p={pval:.3e})")
        if abs(corr) < 0.3:
            print("  ⚠️  WARNING: BW does NOT correlate with distance to cluster center!")
            print("      Adaptive bandwidth is NOT capturing local density correctly.")


if __name__ == "__main__":
    main()
