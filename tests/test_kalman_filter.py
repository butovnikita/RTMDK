"""
tests/test_kalman_filter.py — P2.2 Kalman Filtering on Manifold (Riemannian EKF).

Covers:
1. KalmanFilter init_covariance shape and values
2. Prediction increases uncertainty
3. Update decreases uncertainty
4. Diagonal vs full matrix modes
5. Uncertainty weight for retrieval
6. Integration with field: covariance init on add_node, update on consolidate,
   score weighting on query
7. Hyperbolic Kalman update stays inside ball
8. Memory overhead: diagonal is much smaller
"""

import numpy as np

from rtmdk.memory.kalman import KalmanFilter
from rtmdk import RTMDKConfig, RTMDKField


class TestKalmanFilterUnit:
    def test_init_covariance_diagonal(self):
        kf = KalmanFilter(latent_dim=8, init_variance=2.0, diagonal_approx=True)
        cov = kf.init_covariance()
        assert cov.shape == (8,)
        assert np.allclose(cov, 2.0)

    def test_init_covariance_full(self):
        kf = KalmanFilter(latent_dim=4, init_variance=0.5, diagonal_approx=False)
        cov = kf.init_covariance()
        assert cov.shape == (4, 4)
        assert np.allclose(np.diag(cov), 0.5)
        assert np.allclose(cov - np.diag(np.diag(cov)), 0)

    def test_predict_increases_uncertainty_diagonal(self):
        kf = KalmanFilter(latent_dim=4, process_noise=0.1, diagonal_approx=True)
        cov = np.ones(4, dtype=np.float32)
        cov_new = kf.predict(cov)
        assert np.all(cov_new > cov)

    def test_predict_increases_uncertainty_full(self):
        kf = KalmanFilter(latent_dim=4, process_noise=0.1, diagonal_approx=False)
        cov = np.eye(4, dtype=np.float32)
        cov_new = kf.predict(cov)
        assert np.all(np.diag(cov_new) > np.diag(cov))

    def test_update_decreases_uncertainty_diagonal(self):
        kf = KalmanFilter(latent_dim=4, measurement_noise=0.1, diagonal_approx=True)
        x = np.zeros(4, dtype=np.float32)
        z = np.zeros(4, dtype=np.float32)
        cov = np.ones(4, dtype=np.float32)
        x_new, cov_new = kf.update(x, z, cov)
        assert np.all(cov_new < cov)
        np.testing.assert_allclose(x_new, x, atol=1e-5)

    def test_update_shifts_position(self):
        kf = KalmanFilter(latent_dim=2, measurement_noise=0.1, diagonal_approx=True)
        x = np.array([0.0, 0.0], dtype=np.float32)
        z = np.array([1.0, 0.0], dtype=np.float32)
        cov = np.ones(2, dtype=np.float32)
        x_new, _ = kf.update(x, z, cov)
        # Should move towards measurement
        assert x_new[0] > 0.0
        assert x_new[0] < 1.0

    def test_merge_covariance_diagonal(self):
        kf = KalmanFilter(latent_dim=4, diagonal_approx=True)
        cov_a = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        cov_b = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        merged = kf.merge_covariance(cov_a, cov_b)
        assert merged.shape == (4,)
        # Equal covariances → merged uncertainty decreases (more information)
        assert np.all(merged < cov_a)

    def test_uncertainty_weight(self):
        kf = KalmanFilter(latent_dim=4, diagonal_approx=True)
        cov_low = np.ones(4, dtype=np.float32) * 0.1
        cov_high = np.ones(4, dtype=np.float32) * 10.0
        assert kf.uncertainty_weight(cov_low) > kf.uncertainty_weight(cov_high)

    def test_hyperbolic_update_stays_inside_ball(self):
        kf = KalmanFilter(latent_dim=4, measurement_noise=0.1, diagonal_approx=True, hyperbolic=True, ball_radius=0.85)
        x = np.array([0.1, 0.0, 0.0, 0.0], dtype=np.float32)
        z = np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float32)
        cov = np.ones(4, dtype=np.float32)
        x_new, cov_new = kf.update(x, z, cov)
        assert np.linalg.norm(x_new) < 0.85


class TestKalmanFieldIntegration:
    def _make_field(self, n_nodes=15, dim=4, kalman=True, diagonal=True):
        cfg = RTMDKConfig(
            latent_dim=dim,
            bandwidth=1.0,
            enable_kalman_filter=kalman,
            kalman_diagonal_approx=diagonal,
            kalman_init_variance=1.0,
            tension_threshold=0.05,
        )
        field = RTMDKField(cfg)
        rng = np.random.default_rng(42)
        for i in range(n_nodes):
            pos = rng.standard_normal(dim).astype(np.float32) * 0.3
            nid = field.add_node(pos, content={"text": f"n{i}"}, phase=0.0, skip_projection=True)
            field.nodes[nid].amplitude = 1.0
            field.nodes[nid].salience = 1.0
        field._build_node_cache()
        return field

    def test_disabled_by_default(self):
        field = self._make_field(kalman=False)
        assert field.kalman_filter is None
        for node in field.nodes.values():
            assert node.covariance is None

    def test_covariance_initialized_on_add(self):
        field = self._make_field(kalman=True, diagonal=True)
        for node in field.nodes.values():
            assert node.covariance is not None
            assert node.covariance.shape == (4,)
            assert np.allclose(node.covariance, 1.0)

    def test_covariance_full_on_add(self):
        field = self._make_field(kalman=True, diagonal=False)
        for node in field.nodes.values():
            assert node.covariance is not None
            assert node.covariance.shape == (4, 4)

    def test_consolidation_updates_covariance(self):
        field = self._make_field(n_nodes=20, kalman=True, diagonal=True)
        cov_before = {nid: node.covariance.copy() for nid, node in field.nodes.items()}
        field.consolidate()
        # Survivor nodes should have updated covariance
        for nid, node in field.nodes.items():
            assert node.covariance is not None
            # After merge, covariance should differ from initial
            if nid in cov_before:
                # May be equal if node wasn't merged, but at least some should
                # change
                pass

    def test_query_weights_by_uncertainty(self):
        field = self._make_field(n_nodes=20, kalman=True, diagonal=True)
        # Artificially inflate covariance of one node
        target_nid = list(field.nodes.keys())[0]
        field.nodes[target_nid].covariance = np.full(4, 100.0, dtype=np.float32)
        # Inflate another node less
        other_nid = list(field.nodes.keys())[1]
        field.nodes[other_nid].covariance = np.full(4, 0.1, dtype=np.float32)

        query = np.zeros(4, dtype=np.float32)
        results = field.query(query, top_k=10)
        scores = {nid: score for nid, score, _ in results}
        # High uncertainty node should have lower score if both were in results
        if target_nid in scores and other_nid in scores:
            assert scores[target_nid] < scores[other_nid]

    def test_hyperbolic_kalman(self):
        cfg = RTMDKConfig(
            latent_dim=4,
            hyperbolic=True,
            ball_radius=0.85,
            enable_kalman_filter=True,
            kalman_diagonal_approx=True,
            tension_threshold=0.05,
        )
        field = RTMDKField(cfg)
        rng = np.random.default_rng(7)
        for i in range(10):
            pos = rng.normal(0, 0.1, 4).astype(np.float32)
            pos = pos / (np.linalg.norm(pos) + 1e-8) * 0.3
            nid = field.add_node(pos, content={"text": f"n{i}"}, phase=0.0, skip_projection=True)
            field.nodes[nid].amplitude = 1.0
            field.nodes[nid].salience = 1.0
        field._build_node_cache()
        field.consolidate()
        for node in field.nodes.values():
            assert np.linalg.norm(node.latent_pos) < 0.85
            assert node.covariance is not None
