"""Tests for AdaptiveBandwidthOptimizer."""

import numpy as np

from rtmdk.support.adaptive_bandwidth import AdaptiveBandwidthOptimizer


class TestAdaptiveBandwidthOptimizer:
    def test_init(self):
        opt = AdaptiveBandwidthOptimizer(latent_dim=64)
        assert opt._best_bw is None

    def test_should_optimize_every_n_queries(self):
        opt = AdaptiveBandwidthOptimizer(latent_dim=64, reopt_every=5)
        assert not opt.should_optimize()  # 1
        assert not opt.should_optimize()  # 2
        assert not opt.should_optimize()  # 3
        assert not opt.should_optimize()  # 4
        assert opt.should_optimize()  # 5

    def test_optimize_returns_bandwidth(self):
        opt = AdaptiveBandwidthOptimizer(latent_dim=64, n_candidates=4)
        # 20 random nodes in 64-dim space
        rng = np.random.default_rng(42)
        positions = rng.standard_normal((20, 64)).astype(np.float32)
        positions /= np.linalg.norm(positions, axis=1, keepdims=True) + 1e-8
        phases = np.zeros(20, dtype=np.float32)
        amps = np.ones(20, dtype=np.float32)
        sals = np.ones(20, dtype=np.float32)
        bw = opt.optimize(positions, phases, amps, sals, top_k=5)
        assert bw > 0
        assert opt._best_bw == bw

    def test_optimize_skips_small_sets(self):
        opt = AdaptiveBandwidthOptimizer(latent_dim=64, min_nodes=50)
        positions = np.random.randn(10, 64).astype(np.float32)
        phases = np.zeros(10, dtype=np.float32)
        amps = np.ones(10, dtype=np.float32)
        sals = np.ones(10, dtype=np.float32)
        bw = opt.optimize(positions, phases, amps, sals)
        assert bw == 1.0  # fallback

    def test_state_roundtrip(self):
        opt = AdaptiveBandwidthOptimizer(latent_dim=64)
        opt._best_bw = 2.5
        opt._best_score = 0.8
        state = opt.get_state()
        opt2 = AdaptiveBandwidthOptimizer(latent_dim=64)
        opt2.load_state(state)
        assert opt2._best_bw == 2.5
        assert opt2._best_score == 0.8
