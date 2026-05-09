"""Unit tests for ProjectionManager."""
import numpy as np
import pytest

from rtmdk.memory.projection_manager import ProjectionManager
from rtmdk.memory.config import RTMDKConfig


class TestProjectionManagerInit:
    def test_identity_projection(self):
        cfg = RTMDKConfig(
            embedding_dim=128, latent_dim=128,
            projection_mode="identity", learn_projection=False)
        mgr = ProjectionManager(cfg)
        assert mgr.projection_learner is not None
        assert mgr._raw_projection is None

    def test_random_projection(self):
        cfg = RTMDKConfig(
            embedding_dim=128, latent_dim=64,
            projection_mode="random", learn_projection=False)
        mgr = ProjectionManager(cfg)
        assert mgr.projection_learner is None
        assert mgr._raw_projection is not None
        assert mgr._raw_projection.shape == (128, 64)

    def test_inc_pca_projection(self):
        cfg = RTMDKConfig(
            embedding_dim=128, latent_dim=64,
            projection_mode="pca", learn_projection=True)
        mgr = ProjectionManager(cfg)
        assert mgr.projection_learner is not None
        assert mgr._raw_projection is None

    def test_sot_disabled_by_default(self):
        cfg = RTMDKConfig(sot_enabled=False)
        mgr = ProjectionManager(cfg)
        assert mgr.sot_tokenizer is None
        assert mgr.sot_hebbian is None
        assert mgr._sot_field_ema is None

    def test_sot_enabled(self):
        cfg = RTMDKConfig(
            latent_dim=32, sot_enabled=True,
            sot_max_vocab=100, sot_tokenization_mode="word")
        mgr = ProjectionManager(cfg)
        assert mgr.sot_tokenizer is not None
        assert mgr.sot_hebbian is not None
        assert mgr._sot_field_ema is not None


class TestProjectionManagerProject:
    def test_project_identity(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=64,
            projection_mode="identity", learn_projection=False)
        mgr = ProjectionManager(cfg)
        emb = np.random.randn(64).astype(np.float32)
        latent = mgr.project(emb)
        np.testing.assert_allclose(latent, emb, atol=1e-6)

    def test_project_random(self):
        cfg = RTMDKConfig(
            embedding_dim=128, latent_dim=64,
            projection_mode="random", learn_projection=False)
        mgr = ProjectionManager(cfg)
        emb = np.random.randn(128).astype(np.float64)
        latent = mgr.project(emb)
        assert latent.shape == (64,)
        assert latent.dtype == np.float32

    def test_project_batch(self):
        cfg = RTMDKConfig(
            embedding_dim=128, latent_dim=64,
            projection_mode="random", learn_projection=False)
        mgr = ProjectionManager(cfg)
        embs = np.random.randn(10, 128).astype(np.float32)
        latents = mgr.project_batch(embs)
        assert latents.shape == (10, 64)
        assert latents.dtype == np.float32

    def test_project_batch_identity(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=64,
            projection_mode="identity", learn_projection=False)
        mgr = ProjectionManager(cfg)
        embs = np.random.randn(5, 64).astype(np.float32)
        latents = mgr.project_batch(embs)
        np.testing.assert_allclose(latents, embs, atol=1e-6)

    def test_update_projection_inc_pca(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=32,
            projection_mode="pca", learn_projection=True)
        mgr = ProjectionManager(cfg)
        emb = np.random.randn(64).astype(np.float32)
        latent = mgr.update_projection(emb)
        assert latent.shape == (32,)


class TestProjectionManagerHyperbolic:
    def test_hyperbolic_clamp(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=32,
            projection_mode="random", learn_projection=False,
            hyperbolic=True, ball_radius=1.0)
        mgr = ProjectionManager(cfg)
        emb = np.ones(64, dtype=np.float32) * 10.0
        latent = mgr.project(emb)
        assert np.linalg.norm(latent) < cfg.ball_radius

    def test_hyperbolic_clamp_batch(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=32,
            projection_mode="random", learn_projection=False,
            hyperbolic=True, ball_radius=1.0)
        mgr = ProjectionManager(cfg)
        embs = np.ones((5, 64), dtype=np.float32) * 10.0
        latents = mgr.project_batch(embs)
        norms = np.linalg.norm(latents, axis=1)
        assert np.all(norms < cfg.ball_radius)


class TestProjectionManagerFit:
    def test_fit_projection(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=32,
            projection_mode="pca", learn_projection=True)
        mgr = ProjectionManager(cfg)
        corpus = np.random.randn(100, 64).astype(np.float32)
        mgr.fit_projection(corpus)  # Should not raise

    def test_fit_no_learner(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=32,
            projection_mode="random", learn_projection=False)
        mgr = ProjectionManager(cfg)
        corpus = np.random.randn(100, 64).astype(np.float32)
        mgr.fit_projection(corpus)  # No-op, should not raise


class TestProjectionManagerState:
    def test_get_load_state_identity(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=64,
            projection_mode="identity", learn_projection=False)
        mgr = ProjectionManager(cfg)
        state = mgr.get_state()
        assert "projection_state" in state
        mgr2 = ProjectionManager(cfg)
        mgr2.load_state(state)
        assert mgr2.projection_learner is not None

    def test_get_load_state_random(self):
        cfg = RTMDKConfig(
            embedding_dim=64, latent_dim=32,
            projection_mode="random", learn_projection=False)
        mgr = ProjectionManager(cfg)
        state = mgr.get_state()
        assert "projection" in state
        mgr2 = ProjectionManager(cfg)
        mgr2.load_state(state)
        np.testing.assert_allclose(mgr._raw_projection, mgr2._raw_projection)

    def test_get_load_state_sot(self):
        cfg = RTMDKConfig(
            latent_dim=16, sot_enabled=True,
            sot_max_vocab=50, sot_tokenization_mode="word")
        mgr = ProjectionManager(cfg)
        state = mgr.get_state()
        assert "sot_tokenizer" in state
        mgr2 = ProjectionManager(cfg)
        mgr2.load_state(state)
        assert mgr2.sot_tokenizer is not None


class TestProjectionManagerSOT:
    def test_sot_encode_no_sot(self):
        cfg = RTMDKConfig(sot_enabled=False)
        mgr = ProjectionManager(cfg)
        assert mgr.sot_encode("hello") == []

    def test_sot_query_latent_no_sot(self):
        cfg = RTMDKConfig(sot_enabled=False)
        mgr = ProjectionManager(cfg)
        assert mgr.sot_query_latent("hello") is None

    def test_sot_query_latent_disabled_for_query(self):
        cfg = RTMDKConfig(
            latent_dim=16, sot_enabled=True,
            sot_use_for_query=False)
        mgr = ProjectionManager(cfg)
        assert mgr.sot_query_latent("hello") is None
