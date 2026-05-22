"""
Test different adaptive bandwidth transforms to find one that:
1. Adapts to local density (spread > 2x)
2. Does NOT destroy ranking (R@1 close to baseline)
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


np.random.seed(42)


def generate_challenge_data(n_clusters=20, dim=128):
    """Generate data with HIGH density variation."""
    rng = np.random.default_rng(42)
    embeddings = []
    labels = []
    centers = []
    for i in range(n_clusters):
        center = rng.standard_normal(dim).astype(np.float32)
        center /= np.linalg.norm(center)
        centers.append(center)
        # Varying density: some clusters very tight, some very loose
        if i % 3 == 0:
            n_pts, spread = 80, 0.02   # very dense
        elif i % 3 == 1:
            n_pts, spread = 40, 0.15   # very sparse
        else:
            n_pts, spread = 20, 0.08   # medium
        pts = rng.standard_normal((n_pts, dim)).astype(np.float32) * spread
        pts += center
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        embeddings.append(pts)
        labels.extend([i] * n_pts)
    # Many outliers
    outliers = rng.standard_normal((200, dim)).astype(np.float32)
    outliers = outliers / np.linalg.norm(outliers, axis=1, keepdims=True)
    embeddings.append(outliers)
    labels.extend([-1] * 200)
    return np.vstack(embeddings), np.array(labels), np.array(centers)


def evaluate(field, centers, labels, top_k=5):
    correct_1 = 0
    correct_k = 0
    total = 0
    for ci, center in enumerate(centers):
        results = field.query(center.astype(np.float32), top_k=top_k)
        if not results:
            continue
        top_labels = [labels[int(nid[1:])] for nid, _, _ in results]
        if top_labels[0] == ci:
            correct_1 += 1
        if ci in top_labels:
            correct_k += 1
        total += 1
    return {"R@1": correct_1 / total, "R@5": correct_k / total}


def compute_bw_custom(
    positions,
    global_bw,
    k,
    transform="sqrt",
    clip_range=(
        0.1,
        10.0)):
    """Compute adaptive bw with custom transform."""
    from scipy.spatial import cKDTree
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
    elif transform == "none":
        factors = np.ones_like(kdist)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    factors = np.clip(factors, clip_range[0], clip_range[1])
    return (global_bw * factors).astype(np.float32)


def run_custom(name, X, labels, centers, transform, clip_range):
    cfg = RTMDKConfig(
        latent_dim=128, bandwidth=1.0, adaptive_bandwidth=False,
        resonance_kernel="cosine", phase_coupling=0.0,
        min_response=0.001, use_hnsw=False,
    )
    field = RTMDKField(cfg)
    for i, pos in enumerate(X):
        field.add_node(pos, content={"cluster": int(labels[i])}, phase=0.0,
                       node_id=f"n{i}", skip_projection=True)
        field.nodes[f"n{i}"].amplitude = 1.0
        field.nodes[f"n{i}"].salience = 1.0

    # Force build cache then override _cached_bw with custom computation
    field._build_node_cache()
    # Compute custom bw
    bw = compute_bw_custom(
        field._cached_positions,
        1.0,
        5,
        transform,
        clip_range)
    field._cached_bw = bw

    stats = evaluate(field, centers, labels, top_k=5)
    spread = float(np.percentile(bw, 99) / max(np.percentile(bw, 1), 1e-8))
    print(f"  {name:30s}  R@1={stats['R@1']:.3f}  spread={spread:.1f}x  "
          f"min={bw.min():.3f} max={bw.max():.3f}")
    return stats, spread


def main():
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
    print("Generating challenge dataset (high density variation)...")
    X, labels, centers = generate_challenge_data(n_clusters=20, dim=128)
    print(f"  Total nodes: {len(X)}")

    # Baseline
    cfg = RTMDKConfig(latent_dim=128, bandwidth=1.0, adaptive_bandwidth=False,
                      resonance_kernel="cosine", phase_coupling=0.0,
                      min_response=0.001, use_hnsw=False)
    field = RTMDKField(cfg)
    for i, pos in enumerate(X):
        field.add_node(pos, content={"cluster": int(labels[i])}, phase=0.0,
                       node_id=f"n{i}", skip_projection=True)
        field.nodes[f"n{i}"].amplitude = 1.0
        field.nodes[f"n{i}"].salience = 1.0
    field._build_node_cache()
    base = evaluate(field, centers, labels, top_k=5)
    print(f"\nBaseline (global bw=1.0): R@1={base['R@1']:.3f}")

    print("\n" + "=" * 60)
    print("A/B: Transform + Clip Range")
    print("=" * 60)
    configs = [
        ("sqrt clip[0.2,5.0]", "sqrt", (0.2, 5.0)),
        ("sqrt clip[0.1,10.0]", "sqrt", (0.1, 10.0)),
        ("sqrt clip[0.05,20.0]", "sqrt", (0.05, 20.0)),
        ("linear clip[0.1,10.0]", "linear", (0.1, 10.0)),
        ("linear clip[0.2,5.0]", "linear", (0.2, 5.0)),
        ("log clip[0.1,10.0]", "log", (0.1, 10.0)),
        ("log clip[0.5,3.0]", "log", (0.5, 3.0)),
    ]
    for name, transform, clip_range in configs:
        run_custom(name, X, labels, centers, transform, clip_range)


if __name__ == "__main__":
    main()
