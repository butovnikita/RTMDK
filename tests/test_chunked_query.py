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
        rate_limit_nodes_per_sec=0,
    )
    field = RTMDKField(config)
    rng = np.random.default_rng(42)
    for i in range(n_nodes):
        emb = rng.standard_normal(64).astype(np.float32)
        content = {
            "text": f"node_{i}",
            "tier": "semantic",
            "session": "default"}
        field.add_node(emb, content, phase=rng.random(), session_id="default")
    return field


@pytest.mark.slow
def test_chunked_query_matches_non_chunked():
    """Query with chunking should return same top-k as without chunking."""
    n_nodes = 200
    query_latent = np.random.default_rng(
        7).standard_normal(64).astype(np.float32)
    query_phase = 0.5

    field_small_batch = _make_field(n_nodes, batch_size=50)
    field_large_batch = _make_field(n_nodes, batch_size=500)

    results_small = field_small_batch.query(
        query_latent, top_k=10, phase=query_phase)
    results_large = field_large_batch.query(
        query_latent, top_k=10, phase=query_phase)

    # Extract scores — IDs differ because fields were created at different
    # times (timestamp-based ID generation), but embeddings and query are
    # identical, so scores must match exactly.
    scores_small = [score for nid, score, node in results_small]
    scores_large = [score for nid, score, node in results_large]
    assert len(scores_small) == len(scores_large)
    assert np.allclose(scores_small, scores_large, rtol=1e-5)
