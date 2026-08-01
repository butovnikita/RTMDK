"""Tests for batch ingestion pipeline (Track 4)."""

import json
import numpy as np
import pytest
from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig


@pytest.fixture
def cfg():
    return RTMDKConfig(
        latent_dim=64,
        use_hnsw=True,
        hyperbolic=False,
        bm25_fallback=False,
        quantization="none",
        query_cache_size=0,
        phase_coupling=0.0,
    )


class TestAddNodesBatch:
    def test_batch_returns_correct_ids(self, cfg):
        field = RTMDKField(cfg)
        n = 10
        embeddings = np.random.randn(n, cfg.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        nids = field.add_nodes_batch(embeddings, contents)
        assert len(nids) == n
        assert all(isinstance(nid, str) for nid in nids)

    def test_batch_creates_nodes(self, cfg):
        field = RTMDKField(cfg)
        n = 5
        embeddings = np.random.randn(n, cfg.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        nids = field.add_nodes_batch(embeddings, contents)
        for nid in nids:
            assert nid in field.nodes
        assert len(field.nodes) == n

    def test_batch_hnsw_populated(self, cfg):
        field = RTMDKField(cfg)
        n = 20
        embeddings = np.random.randn(n, cfg.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        field.add_nodes_batch(embeddings, contents)
        if field.hnsw_index:
            assert len(field.hnsw_index.positions) == n

    def test_batch_cache_incremental(self, cfg):
        field = RTMDKField(cfg)
        # Pre-populate to build cache
        field.add_node(np.random.randn(cfg.latent_dim).astype(np.float32), {"text": "seed"})
        # Trigger cache build by a query
        field.query(np.random.randn(cfg.latent_dim).astype(np.float32))
        assert field._cached_positions is not None
        prev_len = field._cached_positions.shape[0]
        n = 5
        embeddings = np.random.randn(n, cfg.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        field.add_nodes_batch(embeddings, contents)
        assert field._cached_positions.shape[0] == prev_len + n

    def test_batch_query_cache_invalidated(self, cfg):
        cfg2 = RTMDKConfig(
            latent_dim=64,
            use_hnsw=True,
            hyperbolic=False,
            bm25_fallback=False,
            quantization="none",
            query_cache_size=100,
        )
        field = RTMDKField(cfg2)
        q = np.random.randn(cfg2.latent_dim).astype(np.float32)
        field.query(q, top_k=3)
        assert field.query_cache.size > 0
        n = 3
        embeddings = np.random.randn(n, cfg2.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        field.add_nodes_batch(embeddings, contents)
        assert field.query_cache.size == 0

    def test_batch_wal_record(self, cfg, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        cfg2 = RTMDKConfig(
            latent_dim=64,
            use_hnsw=True,
            hyperbolic=False,
            bm25_fallback=False,
            quantization="none",
            query_cache_size=0,
        )
        field = RTMDKField(cfg2)
        from rtmdk.memory.wal import WAL

        field.wal = WAL(wal_path)
        n = 4
        embeddings = np.random.randn(n, cfg2.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        nids = field.add_nodes_batch(embeddings, contents)
        with open(wal_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["op"] == "add_nodes_batch"
        assert record["payload"]["count"] == n
        assert set(record["payload"]["node_ids"]) == set(nids)

    def test_batch_equivalent_to_loop(self, cfg):
        """Batch insert should produce identical query results as loop insert."""
        np.random.seed(42)
        field1 = RTMDKField(cfg)
        field2 = RTMDKField(cfg)
        n = 10
        embeddings = np.random.randn(n, cfg.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]

        # Batch
        field1.add_nodes_batch(embeddings, contents)

        # Loop
        nids_loop = []
        for emb, cont in zip(embeddings, contents):
            nid = field2.add_node(emb, cont)
            nids_loop.append(nid)

        # Query both
        q = np.random.randn(cfg.latent_dim).astype(np.float32)
        res1 = field1.query(q, top_k=5)
        res2 = field2.query(q, top_k=5)

        # Same number of results returned
        ids1 = [r[0] for r in res1]
        ids2 = [r[0] for r in res2]
        assert len(ids1) == len(ids2)

        # Same scores (within fp tolerance) — IDs differ because batch vs loop
        # use independent ID generators, but embeddings and resonance are
        # identical.
        scores1 = np.array([r[1] for r in res1])
        scores2 = np.array([r[1] for r in res2])
        np.testing.assert_allclose(scores1, scores2, rtol=1e-5)

    def test_batch_with_modalities(self, cfg):
        cfg2 = RTMDKConfig(
            latent_dim=64,
            use_hnsw=True,
            hyperbolic=False,
            bm25_fallback=False,
            quantization="none",
            query_cache_size=0,
            cross_modal=True,
            modal_phase_offsets={"text": 0.0, "image": 1.0},
        )
        field = RTMDKField(cfg2)
        n = 4
        embeddings = np.random.randn(n, cfg2.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        modalities = ["text", "image", "text", "image"]
        field.add_nodes_batch(embeddings, contents, modalities=modalities)
        for i, nid in enumerate(field.node_index):
            node = field.nodes[nid]
            assert node.modality == modalities[i]

    def test_batch_custom_node_ids(self, cfg):
        field = RTMDKField(cfg)
        n = 3
        embeddings = np.random.randn(n, cfg.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        custom_ids = ["a", "b", "c"]
        nids = field.add_nodes_batch(embeddings, contents, node_ids=custom_ids)
        assert nids == custom_ids
        for nid in custom_ids:
            assert nid in field.nodes

    def test_batch_empty(self, cfg):
        field = RTMDKField(cfg)
        nids = field.add_nodes_batch(np.empty((0, cfg.latent_dim), dtype=np.float32), [])
        assert nids == []

    def test_batch_mismatched_lengths(self, cfg):
        field = RTMDKField(cfg)
        with pytest.raises(ValueError):
            field.add_nodes_batch(np.random.randn(3, cfg.latent_dim).astype(np.float32), [{"text": "x"}])

    def test_batch_with_async_hnsw_and_wal(self, tmp_path):
        wal_path = str(tmp_path / "wal.jsonl")
        cfg = RTMDKConfig(
            latent_dim=64,
            use_hnsw=True,
            hyperbolic=False,
            bm25_fallback=False,
            quantization="none",
            query_cache_size=0,
            async_hnsw_build=True,
            async_hnsw_interval_ms=50,
            async_hnsw_batch_size=1000,
            wal_fsync_interval_ms=50,
            wal_batch_size=10,
        )
        field = RTMDKField(cfg, wal_path=wal_path)
        n = 5
        embeddings = np.random.randn(n, cfg.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        nids = field.add_nodes_batch(embeddings, contents)
        assert len(nids) == n
        # WAL should have the batch record after interval flush
        import time

        time.sleep(0.15)
        records = field.wal.replay()
        assert any(r["op"] == "add_nodes_batch" for r in records)
        # HNSW should be populated after background flush
        assert len(field.hnsw_index.positions) == n
        field.close()

    def test_batch_skip_projection(self, cfg):
        field = RTMDKField(cfg)
        n = 3
        embeddings = np.random.randn(n, cfg.latent_dim).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        nids = field.add_nodes_batch(embeddings, contents, skip_projection=True)
        assert len(nids) == n

    def test_batch_skip_projection_wrong_dim(self, cfg):
        field = RTMDKField(cfg)
        n = 2
        embeddings = np.random.randn(n, cfg.latent_dim + 5).astype(np.float32)
        contents = [{"text": f"node {i}"} for i in range(n)]
        with pytest.raises(ValueError):
            field.add_nodes_batch(embeddings, contents, skip_projection=True)
