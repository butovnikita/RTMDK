"""Test chunked query path in _query_vectorized."""
import time
import numpy as np
import pytest

from rtmdk.memory.core import RTMDKField, RTMDKConfig


def _make_field(n_nodes: int, batch_size: int):
    config = RTMDKConfig(
        latent_dim=64,
        embedding_dim=64,
        use_hnsw=False,
        gpu_batch_size=batch_size,
        bm25_fallback=False,
    )
    field = RTMDKField(config)
    rng = np.random.default_rng(42)
    for i in range(n_nodes):
        emb = rng.standard_normal(64).astype(np.float32)
        content = {"text": f"node_{i}", "tier": "semantic", "session": "default"}
        field.add_node(emb, content, phase=rng.random(), session_id="default")
        time.sleep(0.011)  # stay under 100 nodes/sec rate limit
    return field


def test_chunked_query_matches_non_chunked():
    """Query with chunking should return same top-k as without chunking."""
    n_nodes = 1000
    query_latent = np.random.default_rng(7).standard_normal(64).astype(np.float32)
    query_phase = 0.5

    field_small_batch = _make_field(n_nodes, batch_size=200)
    field_large_batch = _make_field(n_nodes, batch_size=2000)

    results_small = field_small_batch.query(query_latent, top_k=10, phase=query_phase)
    results_large = field_large_batch.query(query_latent, top_k=10, phase=query_phase)

    # Extract IDs and scores
    ids_small = [nid for nid, score, node in results_small]
    ids_large = [nid for nid, score, node in results_large]

    assert ids_small == ids_large, f"Chunked vs non-chunked mismatch: {ids_small} != {ids_large}"

    # Scores should be very close (numerical stability)
    scores_small = [score for nid, score, node in results_small]
    scores_large = [score for nid, score, node in results_large]
    assert np.allclose(scores_small, scores_large, rtol=1e-5)
