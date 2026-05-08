"""Adaptive phase coupling — auto-tune pc based on dataset characteristics."""

import numpy as np
from typing import Optional


def estimate_optimal_pc(
    doc_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    query_targets: Optional[np.ndarray] = None,
    sample_size: int = 50,
) -> float:
    """Estimate optimal phase coupling for a dataset.

    Strategy:
    - Sample queries and compute their top-2 cosine similarity gap.
    - If the gap is large (unambiguous), use low pc (0.0-0.05).
    - If the gap is small (ambiguous), use moderate pc (0.1-0.15).
    - If even top-1 is wrong (very hard), use higher pc (0.2).

    Args:
        doc_embeddings: (N, D) normalized document embeddings
        query_embeddings: (M, D) normalized query embeddings
        query_targets: (M,) integer indices of correct docs for each query
        sample_size: number of queries to sample for estimation

    Returns:
        Recommended phase coupling value in [0.0, 0.3].
    """
    n_queries = len(query_embeddings)
    if n_queries == 0 or len(doc_embeddings) == 0:
        return 0.0

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n_queries, size=min(sample_size, n_queries), replace=False)
    sample_queries = query_embeddings[sample_idx]

    # Cosine similarities: (sample_size, N)
    sims = sample_queries @ doc_embeddings.T

    # Top-2 gaps
    top2 = np.partition(-sims, 1, axis=1)[:, :2]
    top2 = -top2
    gaps = top2[:, 0] - top2[:, 1]
    mean_gap = float(np.mean(gaps))

    # If we have ground truth, compute how often top-1 is correct
    if query_targets is not None:
        sample_targets = query_targets[sample_idx]
        top1_pred = np.argmax(sims, axis=1)
        top1_acc = float(np.mean(top1_pred == sample_targets))
    else:
        top1_acc = None

    # Decision rules (calibrated on comprehensive_500 and qa_1000_en)
    if top1_acc is not None:
        if top1_acc >= 0.95:
            # Dataset is easy — phase coupling adds noise
            return 0.0
        elif top1_acc >= 0.5:
            # Moderately hard — small pc helps cluster paraphrases
            return 0.1
        else:
            # Very hard — higher pc needed for phase clustering
            return 0.15
    else:
        # Fallback: use gap-based heuristic
        if mean_gap >= 0.3:
            return 0.0
        elif mean_gap >= 0.15:
            return 0.1
        else:
            return 0.15
