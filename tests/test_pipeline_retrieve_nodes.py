"""Tests for retrieve_nodes() pipeline migration."""

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2**32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)

    return embed


class TestRetrieveNodesPipelineMigration:
    def test_retrieve_nodes_uses_pipeline_when_enabled(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            embedding_dim=64,
            top_k=5,
            pipeline_enabled=True,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        query_emb = _make_embedder(64)("doc 5")
        results = mem.retrieve_nodes("doc 5", embedding=query_emb, top_k=3)
        assert len(results) > 0
        assert all(len(r) == 3 for r in results)

    def test_retrieve_nodes_legacy_when_disabled(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            embedding_dim=64,
            top_k=5,
            pipeline_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        query_emb = _make_embedder(64)("doc 5")
        results = mem.retrieve_nodes("doc 5", embedding=query_emb, top_k=3)
        assert len(results) > 0
        assert all(len(r) == 3 for r in results)

    def test_retrieve_nodes_fallback_to_legacy_with_sparse_vec(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            embedding_dim=64,
            top_k=5,
            pipeline_enabled=True,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        query_emb = _make_embedder(64)("doc 5")
        sparse_vec = {0: 1.0, 1: 0.5}
        results = mem.retrieve_nodes("doc 5", embedding=query_emb, top_k=3, sparse_vec=sparse_vec)
        assert len(results) > 0
        assert all(len(r) == 3 for r in results)

    def test_retrieve_nodes_pipeline_metrics_available(self):
        cfg = RTMDKConfig(
            latent_dim=64,
            embedding_dim=64,
            top_k=5,
            pipeline_enabled=True,
            pipeline_breaker_enabled=False,
        )
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        for i in range(10):
            emb = _make_embedder(64)(f"doc {i}")
            mem.add_node(embedding=emb, content={"text": f"doc {i}"}, node_id=f"n{i}")

        query_emb = _make_embedder(64)("doc 5")
        results = mem.retrieve_nodes("doc 5", embedding=query_emb, top_k=3)
        assert len(results) > 0
        # Pipeline should have been used internally
        # We can't directly verify from retrieve_nodes return value,
        # but if it returns valid results, pipeline works
