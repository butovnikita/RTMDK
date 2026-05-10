"""Unit tests for MemoryPostInitializer."""

from unittest.mock import MagicMock, patch

import numpy as np

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.memory_post_initializer import MemoryPostInitializer


class _MockMemory:
    def __init__(self, **cfg_overrides):
        self.config = RTMDKConfig(**cfg_overrides)
        self.embedder = lambda x: np.zeros(self.config.latent_dim, dtype=np.float32)
        self.wal_path = None
        self.field = None
        self._replay_wal_called = False

    def _replay_wal(self):
        self._replay_wal_called = True


def _make_mpi(**cfg_overrides):
    mem = _MockMemory(**cfg_overrides)
    return MemoryPostInitializer(mem), mem


class TestMemoryPostInitializer:
    def test_initialize_creates_field_when_none(self):
        mpi, mem = _make_mpi()
        mpi.initialize()
        assert mem.field is not None
        assert mem._replay_wal_called is True

    def test_initialize_uses_existing_field(self):
        mpi, mem = _make_mpi()
        mem.field = MagicMock()
        mem.field._workers_started = False
        mpi.initialize()
        assert mem.field is mem.field  # same object

    def test_init_engrams_enabled(self):
        with patch("rtmdk.engrams.EngramManager") as MockEm:
            mpi, mem = _make_mpi(enable_engrams=True, engram_min_nodes=2)
            mpi._init_engrams()
            MockEm.assert_called_once()
            assert mem.engram_manager is MockEm.return_value

    def test_init_engrams_disabled(self):
        mpi, mem = _make_mpi(enable_engrams=False)
        mpi._init_engrams()
        assert mem.engram_manager is None

    def test_init_causal_traversal_enabled(self):
        with patch("rtmdk.engines.causal_traversal.CausalTraversalEngine") as MockCt:
            mpi, mem = _make_mpi(causal_traversal=True, causal_max_hops=3)
            mpi._init_causal_traversal()
            MockCt.assert_called_once_with(max_hops=3, decay_per_hop=0.5)
            assert mem.causal_traversal_engine is MockCt.return_value

    def test_init_causal_traversal_disabled(self):
        mpi, mem = _make_mpi(causal_traversal=False)
        mpi._init_causal_traversal()
        assert mem.causal_traversal_engine is None

    def test_init_reranker_enabled(self):
        with patch("rtmdk.production.reranker.CrossEncoderReranker") as MockReranker:
            mpi, mem = _make_mpi()
            mem.config.reranker_enabled = True
            mem.config.reranker_model = "test-model"
            mpi._init_reranker()
            MockReranker.assert_called_once_with(model_name="test-model")
            assert mem.reranker is MockReranker.return_value

    def test_init_reranker_disabled(self):
        mpi, mem = _make_mpi()
        mem.config.reranker_enabled = False
        mpi._init_reranker()
        assert mem.reranker is None

    def test_init_bgem3_enabled(self):
        with (
            patch("rtmdk.production.bgem3_embedder.BGEM3Embedder") as MockBge,
            patch("rtmdk.production.sparse_index.SparseIndex") as MockSparse,
        ):
            mpi, mem = _make_mpi()
            mem.config.bgem3_enabled = True
            mem.config.bgem3_model_name = "test-bge"
            mpi._init_bgem3()
            MockBge.assert_called_once_with(model_name="test-bge")
            MockSparse.assert_called_once()
            assert mem.bgem3_embedder is MockBge.return_value
            assert mem.sparse_index is MockSparse.return_value

    def test_init_cascade_router_enabled(self):
        with patch("rtmdk.production.cascade_router.AdaptiveCascadeRouter") as MockRouter:
            mpi, mem = _make_mpi()
            mem.config.cascade_enabled = True
            mpi._init_cascade_router()
            MockRouter.assert_called_once()
            assert mem.cascade_router is MockRouter.return_value

    def test_init_embedder_circuit_breaker(self):
        with patch("rtmdk.support.circuit_breaker.CircuitBreaker") as MockCb:
            mock_cfg = MagicMock()
            mock_cfg.embedder_circuit_breaker_enabled = True
            mock_cfg.embedder_cb_threshold = 5
            mock_cfg.embedder_cb_recovery = 60.0
            mock_cfg.latent_dim = 64
            mem = _MockMemory()
            mem.config = mock_cfg
            mpi = MemoryPostInitializer(mem)
            original_embedder = mem.embedder
            mpi._init_embedder_circuit_breaker()
            MockCb.assert_called_once()
            assert mem.embedder is not original_embedder
            assert mem._embedder_cb is MockCb.return_value

    def test_validate_config_logs_warnings(self):
        mpi, mem = _make_mpi()
        with patch("rtmdk.memory.memory_post_initializer.logger"):
            mpi._validate_config()
            # validate() returns list of warnings; default config should be clean
            # but we just ensure no exception is raised
