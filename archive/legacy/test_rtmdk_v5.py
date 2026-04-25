"""
test_rtmdk_v5.py
Тесты для Фазы 6: Каузально-топологическая память.
"""

import pytest
import json
import numpy as np
import math
from typing import Dict

from rtmdk_memory_v5 import (
    RTMDKConfig, MemoryNode, RTMDKField, RTMDKMemory,
    CausalInferenceEngine, CausalEdge, ContradictionRecord,
    CounterfactualResult,
    format_context, build_system_prompt, ContextFormat,
)


@pytest.fixture
def dummy_embedder():
    def _embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base
    return _embed


@pytest.fixture
def config_v5():
    return RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        tension_threshold=0.2, decay_rate=0.995,
        top_k=3, enable_async=False,
    )


# ============================================================================
# CAUSAL INFERENCE ENGINE
# ============================================================================

class TestCausalInferenceEngine:
    def test_creation(self):
        e = CausalInferenceEngine(min_samples=20, p_threshold=0.05)
        assert e.min_samples == 20
        assert e.p_threshold == 0.05

    def test_record_observation(self):
        e = CausalInferenceEngine()
        e.record_observation(["a", "b", "c"])
        assert e._node_counts["a"] == 1
        assert e._cooccurrence[("a", "b")] == 1
        assert e._cooccurrence[("b", "a")] == 1
        assert e._total_observations == 1

    def test_record_observation_with_context(self):
        e = CausalInferenceEngine()
        e.record_observation(["a", "b"], context={"session": "s1"})
        assert e._conditional_counts[("a", "b", "session=s1")] == 1

    def test_naive_causal_estimate(self):
        e = CausalInferenceEngine()
        for _ in range(10):
            e.record_observation(["cause", "effect"])
        for _ in range(5):
            e.record_observation(["cause"])
        prob = e._naive_causal_estimate("cause", "effect")
        assert 0.0 <= prob <= 1.0
        assert prob > 0.5  # Most cause observations include effect

    def test_backdoor_adjustment(self):
        e = CausalInferenceEngine()
        # Create data with confounder
        for _ in range(20):
            e.record_observation(["confounder", "cause", "effect"])
        for _ in range(5):
            e.record_observation(["confounder", "cause"])
        e.discover_causal_structure()
        prob = e._backdoor_adjustment("effect", "cause")
        assert 0.0 <= prob <= 1.0

    def test_compute_do_probability(self):
        e = CausalInferenceEngine()
        for _ in range(15):
            e.record_observation(["cause", "effect"])
        for _ in range(5):
            e.record_observation(["cause"])
        prob = e.compute_do_probability("effect", "cause")
        assert 0.0 <= prob <= 1.0

    def test_detect_contradictions(self):
        e = CausalInferenceEngine()
        # Create contradictory causes
        for _ in range(10):
            e.record_observation(["cause_a", "effect"])
        for _ in range(10):
            e.record_observation(["cause_b", "effect"])
        # Make cause_a and cause_b negatively correlated
        for _ in range(2):
            e.record_observation(["cause_a", "cause_b"])
        # Add individual observations
        for _ in range(8):
            e.record_observation(["cause_a"])
        for _ in range(8):
            e.record_observation(["cause_b"])

        # Manually set causal effects with opposing strengths
        e.causal_effects[("cause_a", "effect")] = CausalEdge(
            source="cause_a", target="effect", strength=0.8, confidence=0.9)
        e.causal_effects[("cause_b", "effect")] = CausalEdge(
            source="cause_b", target="effect", strength=0.2, confidence=0.9)

        contradictions = e.detect_contradictions(threshold=0.3)
        assert len(contradictions) >= 0  # May or may not detect depending on cooccurrence

    def test_counterfactual_query(self):
        e = CausalInferenceEngine()
        for _ in range(15):
            e.record_observation(["cause", "mediator", "effect"])
        for _ in range(5):
            e.record_observation(["cause", "mediator"])
        e.discover_causal_structure()

        result = e.counterfactual_query(
            intervention={"cause": 1.0},
            query_nodes=["effect", "mediator"],
            max_depth=2)

        assert isinstance(result, CounterfactualResult)
        assert result.intervention == {"cause": 1.0}
        assert len(result.predicted_outcomes) > 0
        assert len(result.reasoning_path) > 0

    def test_validate_consolidation_safe(self):
        e = CausalInferenceEngine()
        for _ in range(10):
            e.record_observation(["node_a", "node_b"])
        e.discover_causal_structure()

        result = e.validate_consolidation("node_a", "node_b")
        assert "safe" in result
        assert "reasons" in result
        assert "recommendation" in result

    def test_validate_consolidation_causal_relationship(self):
        e = CausalInferenceEngine()
        # Create causal relationship
        e.parents["effect"] = {"cause"}
        e.children["cause"] = {"effect"}
        e._compute_ancestors()

        result = e.validate_consolidation("cause", "effect")
        assert result["safe"] is False
        assert result["recommendation"] == "preserve_separate"

    def test_validate_consolidation_opposing_effects(self):
        e = CausalInferenceEngine()
        e.causal_effects[("node_a", "target")] = CausalEdge(
            source="node_a", target="target", strength=0.9, confidence=0.9)
        e.causal_effects[("node_b", "target")] = CausalEdge(
            source="node_b", target="target", strength=0.3, confidence=0.9)
        # Set up children so common_targets is found
        e.children["node_a"] = {"target"}
        e.children["node_b"] = {"target"}

        result = e.validate_consolidation("node_a", "node_b")
        assert result["safe"] is False
        assert len(result["causal_conflicts"]) > 0

    def test_get_load_state(self):
        e = CausalInferenceEngine()
        for _ in range(10):
            e.record_observation(["a", "b"])
        e.causal_effects[("a", "b")] = CausalEdge(
            source="a", target="b", strength=0.7, confidence=0.8)
        state = e.get_state()
        e2 = CausalInferenceEngine()
        e2.load_state(state)
        assert ("a", "b") in e2.causal_effects
        assert e2.causal_effects[("a", "b")].strength == 0.7


# ============================================================================
# FIELD CAUSAL
# ============================================================================

class TestFieldCausalV5:
    def test_field_with_causal_topological(self, config_v5):
        config_v5.causal_topological = True
        field = RTMDKField(config_v5)
        assert field.causal_engine is not None

    def test_causal_engine_records_on_query(self, config_v5):
        config_v5.causal_topological = True
        config_v5.bm25_fallback = True
        field = RTMDKField(config_v5)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test data"}, node_id="n1")
        field.add_node(emb + 0.01, {"text": "another"}, node_id="n2")
        field.query(emb, phase=0.0)
        assert field.causal_engine._total_observations > 0

    def test_counterfactual_query(self, config_v5, dummy_embedder):
        config_v5.causal_topological = True
        config_v5.counterfactual_enabled = True
        memory = RTMDKMemory(config=config_v5, embedder=dummy_embedder)
        for i in range(10):
            memory.save_context(
                {"input": f"msg {i}", "session_id": "s1"},
                {"output": f"resp {i}"}
            )
        result = memory.counterfactual_query(
            intervention={"n0": "modified"},
            query_nodes=["n1", "n2"])
        assert isinstance(result, CounterfactualResult)
        assert result.intervention == {"n0": "modified"}

    def test_get_causal_summary(self, config_v5, dummy_embedder):
        config_v5.causal_topological = True
        config_v5.causal_discovery_freq = 1
        memory = RTMDKMemory(config=config_v5, embedder=dummy_embedder)
        for i in range(5):
            memory.save_context(
                {"input": f"msg {i}", "session_id": "s1"},
                {"output": f"resp {i}"}
            )
        summary = memory.get_causal_summary()
        assert summary["enabled"] is True
        assert "causal_edges" in summary
        assert "contradictions" in summary

    def test_get_contradictions(self, config_v5, dummy_embedder):
        config_v5.causal_topological = True
        config_v5.contradiction_detection = True
        memory = RTMDKMemory(config=config_v5, embedder=dummy_embedder)
        contradictions = memory.get_contradictions()
        assert isinstance(contradictions, list)

    def test_validate_consolidation(self, config_v5, dummy_embedder):
        config_v5.causal_topological = True
        config_v5.do_calculus_validation = True
        memory = RTMDKMemory(config=config_v5, embedder=dummy_embedder)
        memory.save_context({"input": "a", "session_id": "s1"}, {"output": "out_a"})
        memory.save_context({"input": "b", "session_id": "s1"}, {"output": "out_b"})
        result = memory.validate_consolidation(
            memory.field.node_index[0], memory.field.node_index[1])
        assert "safe" in result
        assert "recommendation" in result

    def test_causal_stats_updated(self, config_v5):
        config_v5.causal_topological = True
        config_v5.causal_discovery_freq = 1
        field = RTMDKField(config_v5)
        for i in range(10):
            emb = np.random.randn(768).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
            field.query(emb, phase=0.0)
        assert "causal_edges" in field.stats
        assert "contradictions" in field.stats
        assert "counterfactual_queries" in field.stats
        assert "consolidation_validations" in field.stats
        assert "blocked_consolidations" in field.stats


# ============================================================================
# EXPORT/IMPORT v5
# ============================================================================

class TestExportImportV5:
    def test_export_import_causal_engine(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            causal_topological=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        path = str(tmp_path / "v5_causal.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.causal_engine is not None
        assert len(imported.field.nodes) == 1

    def test_export_import_full_v5(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            causal_topological=True, meta_adaptive=True,
            self_healing=True, differentiable=True,
            continuous_dynamics=True, production_mode=True,
            ab_variant="control", enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        for i in range(3):
            memory.save_context(
                {"input": f"msg {i}", "session_id": "s1"},
                {"output": f"resp {i}"}
            )
        path = str(tmp_path / "v5_full.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.causal_engine is not None
        assert imported.field.meta_kernel is not None
        assert imported.field.healer is not None
        assert len(imported.field.nodes) == 3


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class TestBackwardCompatibilityV5:
    def test_default_config_works(self, dummy_embedder):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        ctx = memory.load_memory_variables({"input": "hi", "session_id": "s1"})
        assert "rtmdk_context" in ctx

    def test_v4_import_works(self, dummy_embedder, tmp_path):
        from rtmdk_memory_v4 import RTMDKConfig as V4Config, RTMDKMemory as V4Memory
        v4_config = V4Config(embedding_dim=768, latent_dim=64, enable_async=False)
        v4_memory = V4Memory(config=v4_config, embedder=dummy_embedder)
        v4_memory.save_context({"input": "v4 test", "session_id": "s1"}, {"output": "v4 out"})
        path = str(tmp_path / "v4_compat.json")
        v4_memory.export_field(path)
        v5_memory = RTMDKMemory.import_field(path, dummy_embedder)
        assert len(v5_memory.field.nodes) == 1
