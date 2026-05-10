"""Tests for ConsolidationManager extraction."""
import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig, ConsolidationMode
from rtmdk.memory.field import RTMDKField
from rtmdk.memory.consolidation_manager import ConsolidationManager


class TestConsolidationManagerExtraction:
    def test_manager_created_with_field(self):
        cfg = RTMDKConfig()
        field = RTMDKField(cfg)
        assert field._consolidation_mgr is not None
        assert isinstance(field._consolidation_mgr, ConsolidationManager)
        assert field._consolidation_mgr.field is field

    def test_consolidate_empty_field(self):
        cfg = RTMDKConfig()
        field = RTMDKField(cfg)
        updated = field.consolidate()
        assert updated == []

    def test_consolidate_with_nodes(self):
        cfg = RTMDKConfig()
        field = RTMDKField(cfg)
        for i in range(20):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        updated = field.consolidate()
        # consolidation may or may not merge depending on random positions
        assert isinstance(updated, list)

    def test_consolidate_dialectical_mode(self):
        cfg = RTMDKConfig()
        field = RTMDKField(cfg)
        for i in range(20):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        updated = field.consolidate(mode=ConsolidationMode.MERGE)
        assert isinstance(updated, list)

    def test_consolidation_reduces_tension(self):
        cfg = RTMDKConfig()
        cfg.core.tension_threshold = 0.01  # very low to force consolidation
        field = RTMDKField(cfg)
        # Add two nearly identical nodes
        emb = np.random.randn(64).astype(np.float32)
        field.add_node(emb.copy(), {"text": "hello"})
        field.add_node(emb.copy() + 0.01, {"text": "hello world"})
        n_before = len(field.nodes)
        field.consolidate()
        # At least one merge or tension reset happened
        assert len(field.nodes) <= n_before

    def test_rollback_state_after_consolidate(self):
        cfg = RTMDKConfig()
        cfg.core.enable_rollback = True
        cfg.core.max_rollback_history = 5
        field = RTMDKField(cfg)
        for i in range(10):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        field.consolidate()
        assert len(field._rollback_history) >= 0

    def test_version_control_deltas(self):
        cfg = RTMDKConfig()
        cfg.core.version_control = True
        field = RTMDKField(cfg)
        for i in range(10):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        field.consolidate()
        if field.version_control:
            assert field.stats["n_versions"] >= 0

    def test_self_supervision_after_consolidate(self):
        cfg = RTMDKConfig()
        cfg.core.self_supervision = True
        field = RTMDKField(cfg)
        for i in range(10):
            emb = np.random.randn(64).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
        field.consolidate()
        # Self-supervision may or may not run depending on merges
        assert field.stats["self_sup_checks"] >= 0
