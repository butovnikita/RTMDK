"""Benchmark KalmanFilter uncertainty weighting on synthetic noisy data.

Tests whether Kalman filtering improves retrieval by down-weighting
high-uncertainty (noisy/stale) nodes.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rtmdk import RTMDKConfig, RTMDKField


def _create_field(kalman: bool, n_nodes: int = 100, dim: int = 32):
    cfg = RTMDKConfig(
        latent_dim=dim,
        enable_kalman_filter=kalman,
        security_enabled=False,  # bypass rate limit for benchmark
        kalman_process_noise=0.01,
        kalman_measurement_noise=0.1,
        kalman_init_variance=0.1,
        kalman_diagonal_approx=True,
        bandwidth=1.0,
    )
    field = RTMDKField(cfg)
    rng = np.random.default_rng(42)

    # Explicit adversarial setup: 4 clusters in corners of a square
    n_clusters = 4
    centers = np.array([
        [0.0, 0.0] + [0.0] * (dim - 2),
        [4.0, 0.0] + [0.0] * (dim - 2),
        [0.0, 4.0] + [0.0] * (dim - 2),
        [4.0, 4.0] + [0.0] * (dim - 2),
    ], dtype=np.float32)
    labels = []
    for i in range(n_nodes):
        cluster_id = i % n_clusters
        pos = centers[cluster_id] + rng.standard_normal(dim).astype(np.float32) * 0.6
        nid = field.add_node(
            pos,
            content={"text": f"node {i} cluster {cluster_id}"},
            phase=0.0,
            node_id=f"n{i}",
            skip_projection=True,
        )
        field.nodes[nid].amplitude = 1.0
        field.nodes[nid].salience = 1.0
        labels.append(cluster_id)
        time.sleep(0.015)

    field._build_node_cache()

    # Adversarial: shift 50% of cluster-0 nodes INTO cluster-1 center
    # This creates many false positives when querying cluster-1
    cluster_0_indices = [i for i, lab in enumerate(labels) if lab == 0]
    noisy_count = len(cluster_0_indices) // 2
    noisy_indices = rng.choice(cluster_0_indices, size=noisy_count, replace=False)
    for idx in noisy_indices:
        nid = f"n{idx}"
        node = field.nodes[nid]
        # Place exactly at cluster-1 center with tiny noise
        node.latent_pos = centers[1] + rng.standard_normal(dim).astype(np.float32) * 0.15
        if kalman:
            node.covariance = np.full(dim, 50.0, dtype=np.float32)

    return field, centers, labels


def _precision(field, centers, labels, top_k: int = 5) -> float:
    """Precision@k: fraction of retrieved nodes that belong to the query cluster."""
    precisions = []
    for cid, center in enumerate(centers):
        results = field.query(center, top_k=top_k)
        retrieved_labels = []
        for nid, score, node in results:
            text = node.content.get("text", "")
            retrieved_labels.append(int(text.split()[-1]))
        hits = sum(1 for lab in retrieved_labels if lab == cid)
        precisions.append(hits / top_k)
    return float(np.mean(precisions))


def main():
    print("=== KalmanFilter Benchmark ===")
    n_nodes = 160
    dim = 32

    field_no_kalman, centers, labels = _create_field(kalman=False, n_nodes=n_nodes, dim=dim)
    field_kalman, _, labels_kalman = _create_field(kalman=True, n_nodes=n_nodes, dim=dim)

    for k in [1, 3, 5]:
        p_no = _precision(field_no_kalman, centers, labels, top_k=k)
        p_kal = _precision(field_kalman, centers, labels_kalman, top_k=k)
        print(f"precision@{k}: no_kalman={p_no:.3f}, kalman={p_kal:.3f}")

    # Primary metric: precision@5
    p_no_kalman = _precision(field_no_kalman, centers, labels, top_k=5)
    p_kalman = _precision(field_kalman, centers, labels_kalman, top_k=5)

    result = {
        "feature": "KalmanFilter",
        "precision@5_no_kalman": round(p_no_kalman, 4),
        "precision@5_kalman": round(p_kalman, 4),
        "improvement_pct": round((p_kalman - p_no_kalman) / (p_no_kalman + 1e-8) * 100, 2),
        "status": "PASS" if p_kalman > p_no_kalman else "FAIL",
        "note": "Kalman should down-weight drifted nodes and improve precision@5",
    }
    print(json.dumps(result, indent=2))

    out = PROJECT_ROOT / "scripts" / "eval_kalman_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
