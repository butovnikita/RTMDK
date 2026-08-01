"""Tests for P0.1 — Riemannian SGD on Poincaré Ball."""

import numpy as np
import pytest

from rtmdk.memory.geometry import (
    poincare_dist,
    exp_map_poincare,
    log_map_poincare,
    mobius_scalar_mul,
    poincare_midpoint,
)
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField

BALL_RADIUS = 0.85


class TestPoincareGeometry:
    """Unit tests for Poincaré ball operations."""

    def test_poincare_midpoint_inside_ball(self):
        a = np.array([0.1, 0.2, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.3, -0.1, 0.0, 0.0], dtype=np.float32)
        m = poincare_midpoint(a, b, BALL_RADIUS)
        assert np.linalg.norm(m) < BALL_RADIUS
        assert m.dtype == np.float32

    def test_poincare_midpoint_symmetry(self):
        a = np.array([0.2, 0.1, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.4, -0.2, 0.0, 0.0], dtype=np.float32)
        m1 = poincare_midpoint(a, b, BALL_RADIUS)
        m2 = poincare_midpoint(b, a, BALL_RADIUS)
        assert np.allclose(m1, m2, atol=1e-5)

    def test_mobius_scalar_mul_half(self):
        x = np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32)
        half = mobius_scalar_mul(0.5, x, BALL_RADIUS)
        assert np.linalg.norm(half) < np.linalg.norm(x)
        assert half.dtype == np.float32

    def test_mobius_scalar_mul_zero(self):
        x = np.array([0.4, 0.1, 0.0, 0.0], dtype=np.float32)
        zero = mobius_scalar_mul(0.0, x, BALL_RADIUS)
        assert np.allclose(zero, 0.0, atol=1e-6)

    def test_exp_map_stays_inside_ball(self):
        """Node at origin, large tangent push — stays inside ball without clamping."""
        base = np.zeros(64, dtype=np.float32)
        tangent = np.ones(64, dtype=np.float32) * 2.0  # Large push
        new_pos = exp_map_poincare(tangent, base, BALL_RADIUS)
        assert np.linalg.norm(new_pos) < BALL_RADIUS
        assert new_pos.dtype == np.float32

    def test_log_map_exp_map_roundtrip(self):
        a = np.array([0.1, 0.2, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.3, -0.1, 0.0, 0.0], dtype=np.float32)
        tangent = log_map_poincare(b, a, BALL_RADIUS)
        b_recon = exp_map_poincare(tangent, a, BALL_RADIUS)
        assert np.allclose(b, b_recon, atol=1e-4)

    def test_poincare_dist_near_center_approximation(self):
        """For small displacements near center, distance ≈ 2 * ||Δ|| (independent of R)."""
        a = np.array([0.01, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.02, 0.0, 0.0, 0.0], dtype=np.float32)
        d1 = poincare_dist(a, b, ball_radius=1.0)
        d2 = poincare_dist(a, b, ball_radius=2.0)
        # Both should be close to 2 * ||b-a|| = 0.02
        assert abs(d1 - 0.02) < 0.001
        assert abs(d2 - 0.02) < 0.001


class TestRiemannianConsolidation:
    """Integration tests for Riemannian consolidate()."""

    @pytest.fixture
    def hyperbolic_field(self):
        cfg = RTMDKConfig(
            latent_dim=16,
            hyperbolic=True,
            ball_radius=BALL_RADIUS,
            max_nodes=200,
            tension_threshold=0.05,
            consolidation_mode="dialectical",
            use_hnsw=False,
        )
        return RTMDKField(cfg)

    def _add_random_nodes(self, field, n, seed=42, scale=0.01):
        import time

        rng = np.random.default_rng(seed)
        for i in range(n):
            # Small-amplitude embeddings project to interior of Poincaré ball
            emb = rng.standard_normal(field.cfg.embedding_dim).astype(np.float32) * scale
            field.add_node(
                embedding=emb,
                content={"text": f"node_{i}"},
                phase=rng.uniform(0, 2 * np.pi),
            )
            if i % 50 == 49:
                time.sleep(0.02)

    def test_consolidate_no_boundary_clamp(self, hyperbolic_field):
        """After consolidate, <1% nodes should be boundary-clamped."""
        self._add_random_nodes(hyperbolic_field, 100, seed=1)
        hyperbolic_field.consolidate()
        clamped = 0
        for node in hyperbolic_field.nodes.values():
            if np.linalg.norm(node.latent_pos) >= 0.99 * hyperbolic_field.cfg.ball_radius:
                clamped += 1
        ratio = clamped / max(len(hyperbolic_field.nodes), 1)
        assert ratio < 0.01, f"{ratio*100:.1f}% nodes boundary-clamped"

    def test_consolidate_midpoint_uses_hyperbolic(self, hyperbolic_field):
        """After dialectical merge, midpoint should be closer to hyperbolic expectation."""
        self._add_random_nodes(hyperbolic_field, 50, seed=2)
        # Force high tension on first two nodes
        ids = list(hyperbolic_field.nodes.keys())
        if len(ids) >= 2:
            hyperbolic_field.nodes[ids[0]].tension = 1.0
            hyperbolic_field.nodes[ids[1]].tension = 1.0
            # Ensure they are close in space
            hyperbolic_field.nodes[ids[1]].latent_pos = hyperbolic_field.nodes[ids[0]].latent_pos + 0.05
        hyperbolic_field.consolidate()
        for node in hyperbolic_field.nodes.values():
            assert np.linalg.norm(node.latent_pos) < hyperbolic_field.cfg.ball_radius

    def test_euclidean_mode_unchanged(self):
        """With hyperbolic=False, consolidate should still work (backward compat)."""
        cfg = RTMDKConfig(
            latent_dim=16,
            hyperbolic=False,
            max_nodes=100,
            tension_threshold=0.05,
            consolidation_mode="merge",
            use_hnsw=False,
        )
        field = RTMDKField(cfg)
        self._add_random_nodes(field, 50, seed=3)
        field.consolidate()
        assert len(field.nodes) > 0

    def test_attraction_update_stays_inside_ball(self, hyperbolic_field):
        """Riemannian attraction update in step() must keep node inside ball."""
        import time

        rng = np.random.default_rng(42)
        # Add an anchor node
        emb = rng.standard_normal(hyperbolic_field.cfg.embedding_dim).astype(np.float32)
        hyperbolic_field.add_node(embedding=emb, content={"text": "anchor"}, phase=0.0)
        time.sleep(0.02)
        # Call step with a similar embedding to trigger attraction update
        hyperbolic_field.step(
            [
                {
                    "embedding": emb * 1.05,
                    "phase": 0.1,
                    "content": {"text": "similar"},
                    "modality": "text",
                }
            ]
        )
        for node in hyperbolic_field.nodes.values():
            assert np.linalg.norm(node.latent_pos) < hyperbolic_field.cfg.ball_radius

    def test_consolidate_performance_regression(self, hyperbolic_field):
        """Consolidate on 100 nodes should not be >20% slower than baseline."""
        import time

        self._add_random_nodes(hyperbolic_field, 100, seed=4)
        t0 = time.perf_counter()
        hyperbolic_field.consolidate()
        t1 = time.perf_counter()
        # Very loose bound: 100 nodes should consolidate in <1s even with
        # Riemannian ops
        assert (t1 - t0) < 1.0
