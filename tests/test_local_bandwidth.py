"""
tests/test_local_bandwidth.py — P1.2 Local Adaptive Bandwidth (k-NN KDE).

Covers:
1. Disabled by default (_cached_bw is None)
2. Computed on cache build (correct length, positive values)
3. Median bandwidth equals global bandwidth (ratio normalization)
4. Different bandwidths for dense vs sparse regions
5. Cache invalidated on node addition
6. Works with session filtering
7. Works in _batch_resonance_numpy
"""

import pytest
import numpy as np

from rtmdk import RTMDKConfig, RTMDKField


def _make_field(n_nodes=10, dim=8, adaptive=True, bw=1.0, **kw):
    """Create a field with deterministic node positions for testing."""
    cfg = RTMDKConfig(
        latent_dim=dim,
        bandwidth=bw,
        adaptive_bandwidth=adaptive,
        adaptive_bandwidth_k=3,
        adaptive_bandwidth_min_n=5,
        **kw
    )
    field = RTMDKField(cfg)
    rng = np.random.default_rng(42)
    for i in range(n_nodes):
        pos = rng.standard_normal(dim).astype(np.float32) * 0.5
        nid = field.add_node(
            pos,
            content={"text": f"node {i}"},
            phase=0.0,
            node_id=f"n{i}",
            skip_projection=True,
        )
        # Force uniform amplitude/salience for predictable resonance tests
        field.nodes[nid].amplitude = 1.0
        field.nodes[nid].salience = 1.0
    field._build_node_cache()
    return field


class TestAdaptiveBandwidthBasics:
    def test_disabled_by_default(self):
        field = _make_field(n_nodes=10, adaptive=False)
        assert field._cached_bw is None

    def test_computed_on_cache_build(self):
        field = _make_field(n_nodes=10, adaptive=True)
        assert field._cached_bw is not None
        assert len(field._cached_bw) == 10
        assert field._cached_bw.dtype == np.float32
        assert np.all(field._cached_bw > 0)

    def test_median_equals_global_bandwidth(self):
        """Median of k-NN distances is the reference → median bw == global bw."""
        field = _make_field(n_nodes=20, adaptive=True, bw=2.5)
        bw_arr = field._cached_bw
        # Due to sqrt(kdist / median(kdist)), the median factor is 1.0
        np.testing.assert_allclose(np.median(bw_arr), 2.5, rtol=0.05)

    def test_not_enough_nodes_fallback(self):
        """If n <= k, adaptive bandwidth is skipped."""
        field = _make_field(n_nodes=3, adaptive=True, bw=1.0)
        # k=3, n=3 → n <= k → fallback to None
        assert field._cached_bw is None

    def test_cache_invalidated_on_add(self):
        field = _make_field(n_nodes=10, adaptive=True)
        assert field._cached_bw is not None
        field.add_node(
            np.zeros(8, dtype=np.float32),
            content={"text": "new"},
            phase=0.0,
            node_id="new",
            skip_projection=True,
        )
        assert field._cache_dirty
        # _cached_bw is cleared immediately in add_node path
        assert field._cached_bw is None

    def test_dense_region_has_smaller_bw(self):
        """Nodes in dense clusters should have smaller bw than outliers."""
        cfg = RTMDKConfig(
            latent_dim=2,
            bandwidth=1.0,
            adaptive_bandwidth=True,
            adaptive_bandwidth_k=3,
            adaptive_bandwidth_min_n=5,
        )
        field = RTMDKField(cfg)
        rng = np.random.default_rng(7)
        # Cluster of 10 nodes near origin
        for i in range(10):
            nid = field.add_node(
                rng.normal(0, 0.05, 2).astype(np.float32),
                content={},
                phase=0.0,
                node_id=f"c{i}",
                skip_projection=True,
            )
            field.nodes[nid].amplitude = 1.0
            field.nodes[nid].salience = 1.0
        # 2 outlier nodes far away
        for i in range(2):
            nid = field.add_node(
                rng.normal(10.0, 0.1, 2).astype(np.float32),
                content={},
                phase=0.0,
                node_id=f"o{i}",
                skip_projection=True,
            )
            field.nodes[nid].amplitude = 1.0
            field.nodes[nid].salience = 1.0
        field._build_node_cache()
        bw = field._cached_bw
        cluster_bw = bw[:10]
        outlier_bw = bw[10:]
        # Outliers have larger k-NN distances → larger bandwidth
        assert np.median(outlier_bw) > np.median(cluster_bw) * 1.5


class TestAdaptiveBandwidthQuery:
    def test_query_vectorized_with_adaptive_bw(self):
        field = _make_field(n_nodes=15, adaptive=True, bw=1.0)
        query = np.zeros(8, dtype=np.float32)
        results = field.query(query, top_k=5)
        assert len(results) == 5
        # Should not raise

    def test_session_filtering_with_adaptive_bw(self):
        cfg = RTMDKConfig(
            latent_dim=8,
            bandwidth=1.0,
            adaptive_bandwidth=True,
            adaptive_bandwidth_k=3,
            adaptive_bandwidth_min_n=5,
        )
        field = RTMDKField(cfg)
        rng = np.random.default_rng(5)
        for i in range(12):
            nid = field.add_node(
                rng.standard_normal(8).astype(np.float32) * 0.3,
                content={"session": "sess_a" if i < 4 else "sess_b"},
                phase=0.0,
                node_id=f"n{i}",
                skip_projection=True,
            )
            field.nodes[nid].amplitude = 1.0
            field.nodes[nid].salience = 1.0
        field._build_node_cache()
        query = np.zeros(8, dtype=np.float32)
        results = field.query(query, top_k=3, session_id="sess_a")
        assert len(results) == 3
        for nid, _, _ in results:
            assert nid.startswith("n0") or nid.startswith("n1") or nid.startswith("n2") or nid.startswith("n3")

    def test_batch_resonance_numpy_uses_per_node_bw(self):
        """_batch_resonance_numpy should index _cached_bw by node id."""
        field = _make_field(n_nodes=12, adaptive=True, bw=1.0)
        query_latents = np.zeros((2, 8), dtype=np.float32)
        query_phases = np.zeros(2, dtype=np.float32)
        node_ids = ["n0", "n1", "n2", "n3"]
        resp = field._batch_resonance_numpy(query_latents, query_phases, node_ids)
        assert resp.shape == (2, 4)
        assert np.all(np.isfinite(resp))

    def test_resonance_differs_from_global(self):
        """With adaptive bw, resonance values should differ from global scalar bw."""
        cfg = RTMDKConfig(latent_dim=4, bandwidth=1.0, adaptive_bandwidth=True, adaptive_bandwidth_k=3, adaptive_bandwidth_min_n=5)
        field = RTMDKField(cfg)
        # Deterministic positions: one cluster, one outlier
        for nid, pos in [
            ("c0", [0.0, 0.0, 0.0, 0.0]),
            ("c1", [0.1, 0.0, 0.0, 0.0]),
            ("c2", [0.0, 0.1, 0.0, 0.0]),
            ("c3", [0.1, 0.1, 0.0, 0.0]),
            ("o0", [5.0, 0.0, 0.0, 0.0]),
        ]:
            field.add_node(
                np.array(pos, dtype=np.float32),
                content={},
                phase=0.0,
                node_id=nid,
                skip_projection=True,
            )
            field.nodes[nid].amplitude = 1.0
            field.nodes[nid].salience = 1.0

        query = np.zeros(4, dtype=np.float32)
        adaptive_results = field.query(query, top_k=5)

        # Same field, global bw
        cfg2 = RTMDKConfig(latent_dim=4, bandwidth=1.0, adaptive_bandwidth=False)
        field2 = RTMDKField(cfg2)
        for nid in field.node_index:
            n = field.nodes[nid]
            field2.add_node(
                n.latent_pos.copy(),
                content={},
                phase=n.phase,
                node_id=nid,
                skip_projection=True,
            )
            field2.nodes[nid].amplitude = n.amplitude
            field2.nodes[nid].salience = n.salience
        global_results = field2.query(query, top_k=5)

        # Ranking or scores should differ (very high probability)
        adaptive_scores = {nid: score for nid, score, _ in adaptive_results}
        global_scores = {nid: score for nid, score, _ in global_results}
        assert adaptive_scores != global_scores
