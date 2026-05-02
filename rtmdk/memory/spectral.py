"""
rtmdk/memory/spectral.py — Spectral Graph Laplacian for consolidation clustering.

Mathematics:
- Affinity: W_ij = exp(-d_ij^2 / 2σ^2) * (1 + cos(φ_i - φ_j)) / 2
- Normalized Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}
- Bottom-k eigenvectors → spectral embedding → k-means
- Eigengap heuristic: k* = argmax_k (λ_{k+1} - λ_k)
"""

import time
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional, Dict


def _build_affinity(positions: NDArray, phases: NDArray, sigma: float = 1.0) -> NDArray:
    """Build affinity matrix using spatial distance and phase coupling."""
    n = positions.shape[0]
    # Pairwise squared distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    # Spatial affinity
    spatial = np.exp(-dists_sq / (2.0 * sigma ** 2))
    # Phase coupling
    phase_diff = np.abs(phases[:, np.newaxis] - phases[np.newaxis, :])
    phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
    phase_coupling = 0.5 + 0.5 * np.cos(phase_diff)
    W = spatial * phase_coupling
    # Zero diagonal
    np.fill_diagonal(W, 0.0)
    return W


def _normalized_laplacian(W: NDArray) -> NDArray:
    """Compute symmetric normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}."""
    d = W.sum(axis=1)
    d_inv_sqrt = np.where(d > 1e-10, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    return L


def _spectral_embedding(L: NDArray, k: int) -> Tuple[NDArray, NDArray]:
    """Compute bottom-k eigenvectors of L.

    Returns (eigenvalues, eigenvectors) sorted ascending by eigenvalue.
    """
    # Use scipy.sparse.linalg.eigsh for larger matrices if available,
    # but for consolidation n is usually < 200, so dense eigh is fine.
    vals, vecs = np.linalg.eigh(L)
    # Sort ascending
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    return vals[:k], vecs[:, :k]


def _eigengap_k(vals: NDArray, max_k: int = 10) -> int:
    """Select k via eigengap heuristic.

    k* = argmax_{k=1..max_k} (λ_{k+1} - λ_k)
    Returns at least 2, at most min(max_k, len(vals)-1).
    """
    m = min(max_k, len(vals) - 1)
    if m < 2:
        return 2
    gaps = vals[1:m+1] - vals[:m]
    # Ignore k=1 if gap is tiny (indicates no structure)
    best_k = int(np.argmax(gaps[1:]) + 2)  # +2 because we skip k=1
    if best_k < 2:
        best_k = 2
    return min(best_k, m)


def _kmeans_1d(embeddings: NDArray, k: int, max_iter: int = 20, rng: Optional[np.random.Generator] = None) -> NDArray:
    """Simple k-means on spectral embedding (k is small, dims <= k).

    Returns cluster labels (0..k-1).
    """
    n, dim = embeddings.shape
    rng = rng or np.random.default_rng()
    # k-means++ initialization
    centers = np.zeros((k, dim))
    centers[0] = embeddings[rng.integers(n)]
    for i in range(1, k):
        dists = np.min(np.sum((embeddings[:, np.newaxis, :] - centers[np.newaxis, :i, :]) ** 2, axis=2), axis=1)
        probs = dists / (dists.sum() + 1e-10)
        centers[i] = embeddings[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assignment
        dists = np.sum((embeddings[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centers
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = embeddings[mask].mean(axis=0)
    return labels


def spectral_cluster_nodes(
    positions: NDArray,
    phases: NDArray,
    max_clusters: int = 10,
    sigma: float = 1.0,
    timeout_ms: float = 500.0,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Dict[int, List[int]]]:
    """Cluster nodes using spectral graph Laplacian.

    Args:
        positions: (N, d) node positions
        phases: (N,) node phases
        max_clusters: upper bound on number of clusters
        sigma: bandwidth for affinity computation
        timeout_ms: abort and return None if exceeded
        rng: random generator for k-means init

    Returns:
        dict mapping cluster_id -> list of node indices, or None on failure/timeout.
    """
    t0 = time.time()
    n = positions.shape[0]
    if n < 3:
        return None

    try:
        W = _build_affinity(positions, phases, sigma)
        if (time.time() - t0) * 1000 > timeout_ms:
            return None

        L = _normalized_laplacian(W)
        if (time.time() - t0) * 1000 > timeout_ms:
            return None

        # Compute up to max_clusters+1 eigenvalues for eigengap
        k_eig = min(max_clusters + 1, n)
        vals, vecs = _spectral_embedding(L, k_eig)
        if (time.time() - t0) * 1000 > timeout_ms:
            return None

        k = _eigengap_k(vals, max_clusters)
        # Row-normalize embedding (standard spectral clustering)
        embedding = vecs[:, :k]
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        embedding = embedding / norms

        labels = _kmeans_1d(embedding, k, rng=rng)

        clusters: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels):
            clusters.setdefault(int(lab), []).append(idx)
        return clusters
    except Exception:
        return None
