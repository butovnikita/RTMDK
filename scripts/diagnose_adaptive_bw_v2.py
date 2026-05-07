"""
Diagnostic v2: Test adaptive_bw with RANDOM PROJECTION (realistic pipeline).

Hypothesis: Random projection 768d -> 128d distorts local density,
causing extreme bw factors that destroy ranking.
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


np.random.seed(42)


def generate_realistic_embeddings(n_clusters=20, dim=768):
    """Generate realistic embeddings:
    - Some clusters are tight (high density), some are loose
    - Some clusters have many points, some few
    - Outliers scattered
    - All normalized (cosine space, like real embeddings)
    """
    rng = np.random.default_rng(42)
    embeddings = []
    labels = []
    centers = []
    cluster_info = []

    for i in range(n_clusters):
        # Varying cluster properties
        n_pts = rng.integers(20, 100)
        spread = rng.uniform(0.02, 0.15)  # tight vs loose
        center = rng.standard_normal(dim).astype(np.float32)
        center /= np.linalg.norm(center)
        centers.append(center)
        cluster_info.append((n_pts, spread))

        pts = rng.standard_normal((n_pts, dim)).astype(np.float32) * spread
        pts += center
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        embeddings.append(pts)
        labels.extend([i] * n_pts)

    # Outliers: random on sphere
    n_outliers = 100
    outliers = rng.standard_normal((n_outliers, dim)).astype(np.float32)
    outliers = outliers / np.linalg.norm(outliers, axis=1, keepdims=True)
    embeddings.append(outliers)
    labels.extend([-1] * n_outliers)

    X = np.vstack(embeddings)
    return X, np.array(labels), np.array(centers), cluster_info


def build_field(X, labels, cfg, skip_proj=False):
    field = RTMDKField(cfg)
    for i, pos in enumerate(X):
        field.add_node(
            pos.astype(np.float32),
            content={"cluster": int(labels[i])},
            phase=0.0,
            node_id=f"n{i}",
            skip_projection=skip_proj,
        )
        field.nodes[f"n{i}"].amplitude = 1.0
        field.nodes[f"n{i}"].salience = 1.0
    return field


def evaluate(field, centers, labels, top_k=5):
    correct_at_1 = 0
    correct_at_k = 0
    total = 0
    for ci, center in enumerate(centers):
        results = field.query(center.astype(np.float32), top_k=top_k)
        if not results:
            continue
        top_ids = [nid for nid, _, _ in results]
        top_labels = [labels[int(nid[1:])] for nid in top_ids]
        if top_labels[0] == ci:
            correct_at_1 += 1
        if ci in top_labels:
            correct_at_k += 1
        total += 1
    return {
        "R@1": correct_at_1 / total if total else 0,
        "R@5": correct_at_k / total if total else 0,
    }


def bw_stats(field):
    if field._cached_bw is None:
        return None
    bw = field._cached_bw
    return {
        "mean": float(
            np.mean(bw)), "median": float(
            np.median(bw)), "std": float(
                np.std(bw)), "min": float(
                    np.min(bw)), "max": float(
                        np.max(bw)), "p01": float(
                            np.percentile(
                                bw, 1)), "p99": float(
                                    np.percentile(
                                        bw, 99)), "ratio": float(
                                            np.percentile(
                                                bw, 99) / max(
                                                    np.percentile(
                                                        bw, 1), 1e-8)), }


def run(name, cfg, X, labels, centers, skip_proj=False):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")
    field = build_field(X, labels, cfg, skip_proj=skip_proj)
    field._build_node_cache()
    stats = evaluate(field, centers, labels, top_k=5)
    bw = bw_stats(field)
    print(f"  R@1: {stats['R@1']:.3f}  R@5: {stats['R@5']:.3f}")
    if bw:
        print(
            f"  BW  mean={bw['mean']:.3f} med={bw['median']:.3f} std={bw['std']:.3f}")
        print(
            f"  BW  min={bw['min']:.4f} max={bw['max']:.4f} p01={bw['p01']:.4f} p99={bw['p99']:.4f}")
        print(f"  BW  p99/p01 ratio: {bw['ratio']:.2f}x")
    return field, stats, bw


def main():
    print("Generating realistic synthetic embeddings (768d, cosine-normalized)...")
    X, labels, centers, cluster_info = generate_realistic_embeddings(
        n_clusters=20, dim=768)
    print(f"  Total nodes: {len(X)}  Clusters: 20  Outliers: 100")
    print(f"  Cluster spreads: {[f'{s:.2f}' for _, s in cluster_info[:5]]}...")

    # --- No projection ---
    cfg_no_proj = RTMDKConfig(
        latent_dim=768,
        bandwidth=1.0,
        adaptive_bandwidth=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f1, s1, bw1 = run("No projection, global bw", cfg_no_proj,
                      X, labels, centers, skip_proj=True)

    cfg_adapt_no_proj = RTMDKConfig(
        latent_dim=768,
        bandwidth=1.0,
        adaptive_bandwidth=True,
        adaptive_bandwidth_k=5,
        adaptive_bandwidth_min_n=5,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f2, s2, bw2 = run("No projection, adaptive k=5",
                      cfg_adapt_no_proj, X, labels, centers, skip_proj=True)

    # --- With random projection 768->128 ---
    cfg_proj = RTMDKConfig(
        latent_dim=128,
        bandwidth=1.0,
        adaptive_bandwidth=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
        learn_projection=False,  # use random projection matrix
    )
    f3, s3, bw3 = run("Projection 768->128, global bw",
                      cfg_proj, X, labels, centers, skip_proj=False)

    cfg_adapt_proj = RTMDKConfig(
        latent_dim=128,
        bandwidth=1.0,
        adaptive_bandwidth=True,
        adaptive_bandwidth_k=5,
        adaptive_bandwidth_min_n=5,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
        learn_projection=False,
    )
    f4, s4, bw4 = run("Projection 768->128, adaptive k=5",
                      cfg_adapt_proj, X, labels, centers, skip_proj=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    rows = [
        ("No proj, global", s1, bw1),
        ("No proj, adaptive", s2, bw2),
        ("Proj, global", s3, bw3),
        ("Proj, adaptive", s4, bw4),
    ]
    for name, st, bw in rows:
        extra = f"  bw_ratio={bw['ratio']:.1f}x" if bw else ""
        print(f"  {name:20s}  R@1={st['R@1']:.3f}  R@5={st['R@5']:.3f}{extra}")

    # --- Diagnostic: does projection create extreme bw factors? ---
    if bw4:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC: BW factor distribution")
        print("=" * 60)
        print(f"  Global bw = 1.0")
        print(f"  Adaptive bw range: {bw4['min']:.4f} to {bw4['max']:.4f}")
        print(f"  Spread ratio (p99/p01): {bw4['ratio']:.1f}x")
        if bw4['ratio'] > 50:
            print("  ⚠️  EXTREME spread! Some nodes have 50x+ bandwidth vs others.")
            print("      This will destroy fine-grained ranking.")
        elif bw4['ratio'] > 10:
            print("  ⚠️  Large spread (>10x). May degrade ranking on boundary cases.")
        else:
            print("  ✓  Moderate spread. Ranking should be stable.")


if __name__ == "__main__":
    main()
