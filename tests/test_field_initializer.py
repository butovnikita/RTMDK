"""Unit tests for FieldInitializer."""

import numpy as np
import pytest

from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig


class TestFieldInitializer:
    def test_field_has_critical_attributes(self):
        """After initialization, RTMDKField must expose all attributes that managers depend on."""
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        field = RTMDKField(cfg)

        # Core data structures
        assert hasattr(field, "nodes")
        assert hasattr(field, "node_index")
        assert hasattr(field, "cfg")
        assert hasattr(field, "stats")

        # Quantization & RNG
        assert hasattr(field, "_quant")
        assert hasattr(field, "_rng")

        # Managers (must exist for delegation)
        assert hasattr(field, "_node_mgr")
        assert hasattr(field, "_query_mgr")
        assert hasattr(field, "_topology_mgr")
        assert hasattr(field, "_async_pipeline_mgr")
        assert hasattr(field, "_crystallization_mgr")
        assert hasattr(field, "_merge_mgr")
        assert hasattr(field, "_routing_mgr")
        assert hasattr(field, "_index_mgr")
        assert hasattr(field, "_projection_mgr")
        assert hasattr(field, "_consolidation_mgr")
        assert hasattr(field, "_cognitive_mgr")
        assert hasattr(field, "_operational_mgr")
        assert hasattr(field, "_scheduler")
        assert hasattr(field, "_cache_mgr")

        # Engines / helpers
        assert hasattr(field, "_resonance_engine")
        assert hasattr(field, "wal")

    def test_field_nodes_is_empty_dict(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        field = RTMDKField(cfg)
        assert field.nodes == {}
        assert field.node_index == []

    def test_field_stats_initialized(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        field = RTMDKField(cfg)
        assert isinstance(field.stats, dict)
        assert "total_adds" in field.stats

    def test_field_with_wal_path(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_path = os.path.join(tmpdir, "test.wal")
            cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
            field = RTMDKField(cfg, wal_path=wal_path)
            assert field.wal is not None
            assert field.wal.path == wal_path
            field.close()

    def test_field_quantization_helper_mode(self):
        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64, quantization="none")
        field = RTMDKField(cfg)
        assert field._quant.mode == "none"
        field.close()
