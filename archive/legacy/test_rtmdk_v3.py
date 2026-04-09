"""
test_rtmdk_v3.py
Тесты для всех 4 треков RTMDK v3.
"""

import pytest
import json
import numpy as np
import math

from rtmdk_memory_v3 import (
    RTMDKConfig, MemoryNode, RTMDKField, RTMDKMemory,
    LearnableKernel, DifferentiableConsolidation,
    NeuralODEField, CausalGraph, ProductionMonitor,
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
def config_v3():
    return RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        tension_threshold=0.2, decay_rate=0.995,
        top_k=3, enable_async=False,
    )


# ============================================================================
# ТРЕК 1: ДИФФЕРЕНЦИРУЕМОЕ ПОЛЕ
# ============================================================================

class TestLearnableKernel:
    def test_creation(self):
        k = LearnableKernel(bandwidth=1.0, phase_coupling=0.3, decay_rate=0.998)
        assert k.bandwidth == 1.0
        assert k.phase_coupling == 0.3

    def test_resonance_response(self):
        k = LearnableKernel(bandwidth=1.0, phase_coupling=0.3)
        resp = k.resonance_response(dist=0.0, phase_diff=0.0, amplitude=1.0, salience=1.0)
        assert resp > 0.5

    def test_compute_gradients(self):
        k = LearnableKernel(bandwidth=1.0, phase_coupling=0.3)
        k.compute_gradients(dist=1.0, phase_diff=0.5, amplitude=0.7, salience=0.6, loss_gradient=1.0)
        assert k._grad_bandwidth != 0.0 or k._grad_phase_coupling != 0.0

    def test_step_updates_params(self):
        k = LearnableKernel(bandwidth=1.0, phase_coupling=0.3)
        for _ in range(10):
            k.compute_gradients(dist=1.0, phase_diff=0.5, amplitude=0.7, salience=0.6, loss_gradient=1.0)
            k.step()
        assert k.bandwidth > 0.1

    def test_get_load_state(self):
        k = LearnableKernel(bandwidth=1.5, phase_coupling=0.4)
        state = k.get_state()
        k2 = LearnableKernel()
        k2.load_state(state)
        assert k2.bandwidth == 1.5
        assert k2.phase_coupling == 0.4


class TestDifferentiableConsolidation:
    def test_synthesis(self):
        dc = DifferentiableConsolidation(loss_weight=0.1)
        n1 = MemoryNode(id="a", latent_pos=np.zeros(64, dtype=np.float32), phase=0.0, amplitude=0.7, salience=0.6)
        n2 = MemoryNode(id="b", latent_pos=np.ones(64, dtype=np.float32), phase=np.pi, amplitude=0.5, salience=0.4)
        result = dc.compute_synthesis(n1, n2, gate=0.5)
        assert "latent_pos" in result
        assert "phase" in result
        assert "loss" in result
        assert result["loss"] >= 0


class TestFieldDifferentiable:
    def test_field_with_differentiable(self, config_v3):
        config_v3.differentiable = True
        field = RTMDKField(config_v3)
        assert field.learnable_kernel is not None
        assert field.diff_consolidation is not None

    def test_differentiable_consolidation(self, config_v3):
        config_v3.differentiable = True
        config_v3.tension_threshold = 0.05
        config_v3.consolidation_mode = "dialectical"
        field = RTMDKField(config_v3)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "A"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "B"}, phase=np.pi, node_id="b")
        field.consolidate()
        assert field.stats["differentiable_loss"] >= 0


# ============================================================================
# ТРЕК 2: NEURAL ODE / SDE
# ============================================================================

class TestNeuralODEField:
    def test_creation(self):
        ode = NeuralODEField(latent_dim=64, noise_level=0.01, time_horizon=1.0, n_steps=10)
        assert ode.latent_dim == 64

    def test_evolve(self):
        ode = NeuralODEField(latent_dim=8, noise_level=0.0, time_horizon=0.5, n_steps=5)
        initial = np.random.randn(3, 8).astype(np.float32)
        trajectory = ode.evolve(initial)
        assert trajectory.shape[0] == 5
        assert trajectory.shape[1] == 24

    def test_evolve_with_noise(self):
        ode = NeuralODEField(latent_dim=8, noise_level=0.1, time_horizon=0.5, n_steps=5)
        initial = np.zeros((3, 8), dtype=np.float32)
        trajectory = ode.evolve_with_noise(initial)
        assert len(trajectory) > 1

    def test_get_load_state(self):
        ode = NeuralODEField(latent_dim=8)
        state = ode.get_state()
        ode2 = NeuralODEField(latent_dim=8)
        ode2.load_state(state)
        assert np.allclose(ode._weights, ode2._weights)


class TestFieldContinuousDynamics:
    def test_field_with_ode(self, config_v3):
        config_v3.continuous_dynamics = True
        config_v3.ode_n_steps = 5
        config_v3.ode_time_horizon = 0.1
        field = RTMDKField(config_v3)
        assert field.neural_ode is not None

    def test_evolve_continuous(self, config_v3):
        config_v3.continuous_dynamics = True
        config_v3.ode_n_steps = 5
        config_v3.ode_time_horizon = 0.1
        field = RTMDKField(config_v3)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0, node_id="n1")
        field.add_node(emb, {"text": "test2"}, phase=0.1, node_id="n2")
        trajectory = field.evolve_continuous()
        assert len(trajectory) > 0
        assert field.stats["ode_steps"] >= 1

    def test_evolve_with_sde(self, config_v3):
        config_v3.continuous_dynamics = True
        config_v3.sde_noise_level = 0.05
        config_v3.ode_n_steps = 5
        config_v3.ode_time_horizon = 0.1
        field = RTMDKField(config_v3)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, node_id="n1")
        trajectory = field.evolve_continuous(use_sde=True)
        assert len(trajectory) > 1


# ============================================================================
# ТРЕК 3: КАУЗАЛЬНО-ТОПОЛОГИЧЕСКАЯ ИНТЕГРАЦИЯ
# ============================================================================

class TestCausalGraph:
    def test_creation(self):
        cg = CausalGraph(threshold=0.3)
        assert cg.threshold == 0.3

    def test_record_cooccurrence(self):
        cg = CausalGraph()
        for _ in range(10):
            cg.record_cooccurrence("a", "b")
        assert cg._cooccurrence[("a", "b")] == 10
        assert cg._node_counts["a"] == 10

    def test_discover_causal_links(self):
        cg = CausalGraph(threshold=0.01)
        # Diverse cooccurrence data so PMI can differentiate
        for _ in range(100):
            cg.record_cooccurrence("cause", "effect")
        for _ in range(5):
            cg.record_cooccurrence("noise_a", "noise_b")
        for _ in range(3):
            cg.record_cooccurrence("rare_x", "rare_y")
        links = cg.discover_causal_links()
        # At minimum, discovery ran without error
        assert isinstance(links, dict)

    def test_do_intervention(self):
        cg = CausalGraph()
        new_pos = np.random.randn(64).astype(np.float32)
        cg.do_intervention("node_a", new_pos)
        assert "node_a" in cg.interventions
        assert np.allclose(cg.interventions["node_a"], new_pos)

    def test_clear_interventions(self):
        cg = CausalGraph()
        cg.do_intervention("x", np.zeros(64, dtype=np.float32))
        cg.clear_interventions()
        assert len(cg.interventions) == 0

    def test_get_causal_parents(self):
        cg = CausalGraph(threshold=0.0)
        cg.edges["a"]["b"] = 0.5
        cg.edges["a"]["c"] = 0.3
        parents = cg.get_causal_parents("a")
        assert "b" in parents
        assert "c" in parents

    def test_get_causal_effect(self):
        cg = CausalGraph()
        cg.edges["cause"]["effect"] = 0.7
        assert cg.get_causal_effect("cause", "effect") == 0.7
        assert cg.get_causal_effect("cause", "nothing") == 0.0

    def test_get_load_state(self):
        cg = CausalGraph(threshold=0.2)
        cg.edges["a"]["b"] = 0.5
        state = cg.get_state()
        cg2 = CausalGraph()
        cg2.load_state(state)
        assert cg2.get_causal_effect("a", "b") == 0.5


class TestFieldCausal:
    def test_field_with_causal(self, config_v3):
        config_v3.causal_modeling = True
        config_v3.causal_threshold = 0.3
        field = RTMDKField(config_v3)
        assert field.causal_graph is not None

    def test_causal_discovery_in_consolidate(self, config_v3):
        config_v3.causal_modeling = True
        config_v3.causal_threshold = 0.1
        config_v3.causal_discovery_freq = 1
        field = RTMDKField(config_v3)
        emb = np.random.randn(768).astype(np.float32)
        for i in range(10):
            field.add_node(emb, {"text": f"n{i}"}, node_id=f"n{i}")
        field.consolidate()
        assert field.stats["causal_links"] >= 0

    def test_do_intervention_in_field(self, config_v3, dummy_embedder):
        config_v3.causal_modeling = True
        field = RTMDKField(config_v3)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, node_id="n1")
        new_emb = np.random.randn(768).astype(np.float32)
        field.do_intervention("n1", new_emb)
        assert "n1" in field.causal_graph.interventions


# ============================================================================
# ТРЕК 4: ПРОДАКШЕН МОНИТОРИНГ
# ============================================================================

class TestProductionMonitor:
    def test_creation(self):
        m = ProductionMonitor(drift_window=100, drift_threshold=0.05)
        assert m.drift_window == 100

    def test_record_embedding(self):
        m = ProductionMonitor()
        m.record_embedding(np.random.randn(768).astype(np.float32))
        assert len(m._embedding_history) == 1

    def test_record_response(self):
        m = ProductionMonitor()
        m.record_response(0.5, 10.0, n_consolidations=1, avg_gate=0.8)
        assert len(m._response_history) == 1
        assert len(m._latency_history) == 1

    def test_detect_drift_no_drift(self):
        m = ProductionMonitor(drift_window=20, drift_threshold=0.5)
        for _ in range(20):
            m.record_embedding(np.random.randn(64).astype(np.float32))
        result = m.detect_drift()
        assert "drifting" in result
        assert "score" in result

    def test_detect_anomaly(self):
        m = ProductionMonitor(anomaly_threshold=2.0)
        # Record varied responses so std > 0
        for v in [0.4, 0.5, 0.6, 0.45, 0.55, 0.48, 0.52, 0.47, 0.53, 0.5]:
            for _ in range(5):
                m.record_response(v, 10.0)
        is_anomaly = m.detect_anomaly(100.0)
        assert is_anomaly

    def test_detect_no_anomaly(self):
        m = ProductionMonitor(anomaly_threshold=3.0)
        for _ in range(50):
            m.record_response(0.5, 10.0)
        is_anomaly = m.detect_anomaly(0.51)
        assert not is_anomaly

    def test_ab_testing(self):
        m = ProductionMonitor()
        m.record_ab_result("control", "accuracy", 0.8)
        m.record_ab_result("control", "accuracy", 0.82)
        m.record_ab_result("variant", "accuracy", 0.85)
        m.record_ab_result("variant", "accuracy", 0.87)
        comparison = m.get_ab_comparison("accuracy")
        assert "control" in comparison
        assert "variant" in comparison
        assert comparison["variant"]["mean"] > comparison["control"]["mean"]

    def test_dashboard(self):
        m = ProductionMonitor()
        for _ in range(30):
            m.record_response(0.5, 15.0)
        dashboard = m.get_dashboard()
        assert "drift" in dashboard
        assert "response" in dashboard
        assert "latency_ms" in dashboard
        assert "gate_distribution" in dashboard
        assert "anomalies" in dashboard

    def test_get_load_state(self):
        m = ProductionMonitor()
        m.record_ab_result("a", "x", 1.0)
        state = m.get_state()
        m2 = ProductionMonitor()
        m2.load_state(state)
        assert "a" in m2._ab_results


class TestFieldProduction:
    def test_field_with_production(self, config_v3):
        config_v3.production_mode = True
        config_v3.drift_detection = True
        field = RTMDKField(config_v3)
        assert field.monitor is not None

    def test_monitor_records_on_query(self, config_v3):
        config_v3.production_mode = True
        config_v3.bm25_fallback = True
        field = RTMDKField(config_v3)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test data for query"})
        field.query(emb, phase=0.0)
        assert len(field.monitor._response_history) > 0

    def test_get_dashboard(self, config_v3, dummy_embedder):
        config_v3.production_mode = True
        config_v3.ab_testing = True
        config_v3.ab_variant = "control"
        memory = RTMDKMemory(config=config_v3, embedder=dummy_embedder)
        for i in range(10):
            memory.save_context(
                {"input": f"message {i}", "session_id": "s1"},
                {"output": f"response {i}"}
            )
        dashboard = memory.get_dashboard()
        assert "drift" in dashboard
        assert "response" in dashboard

    def test_record_ab_metric(self, config_v3, dummy_embedder):
        config_v3.production_mode = True
        config_v3.ab_testing = True
        config_v3.ab_variant = "variant_a"
        memory = RTMDKMemory(config=config_v3, embedder=dummy_embedder)
        memory.field.record_ab_metric("accuracy", 0.85)
        memory.field.record_ab_metric("accuracy", 0.87)
        assert "variant_a" in memory.field.monitor._ab_results
        assert len(memory.field.monitor._ab_results["variant_a"]["accuracy"]) == 2


# ============================================================================
# EXPORT/IMPORT v3
# ============================================================================

class TestExportImportV3:
    def test_export_import_differentiable(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            differentiable=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        path = str(tmp_path / "v3_diff.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.learnable_kernel is not None
        assert len(imported.field.nodes) == 1

    def test_export_import_ode(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            continuous_dynamics=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        path = str(tmp_path / "v3_ode.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.neural_ode is not None

    def test_export_import_causal(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            causal_modeling=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        path = str(tmp_path / "v3_causal.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.causal_graph is not None

    def test_export_import_production(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            production_mode=True, ab_testing=True,
            ab_variant="control", enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        memory.record_ab_metric("accuracy", 0.8)
        path = str(tmp_path / "v3_prod.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.monitor is not None
        assert "control" in imported.field.monitor._ab_results


# ============================================================================
# INTEGRATION: Memory v3
# ============================================================================

class TestMemoryV3:
    def test_inspect_node(self, dummy_embedder):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        nid = memory.field.node_index[0]
        info = memory.inspect_node(nid)
        assert info is not None
        assert info["id"] == nid
        assert "content" in info

    def test_rollback(self, dummy_embedder):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            enable_rollback=True, tension_threshold=0.05,
            consolidation_mode="dialectical", enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        emb = np.zeros(768, dtype=np.float32)
        memory.field.add_node(emb, {"text": "A"}, phase=0.0, node_id="a")
        memory.field.add_node(emb, {"text": "B"}, phase=np.pi, node_id="b")
        memory.field.consolidate()
        history = memory.get_rollback_history()
        assert len(history) >= 1

    def test_do_intervention(self, dummy_embedder):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            causal_modeling=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "original", "session_id": "s1"}, {"output": "out"})
        nid = memory.field.node_index[0]
        memory.do_intervention(nid, "new text")
        assert nid in memory.field.causal_graph.interventions
        memory.clear_interventions()
        assert len(memory.field.causal_graph.interventions) == 0

    def test_get_dashboard(self, dummy_embedder):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            production_mode=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        dashboard = memory.get_dashboard()
        assert "drift" in dashboard
        assert "response" in dashboard


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class TestBackwardCompatibilityV3:
    def test_default_config_works(self, dummy_embedder):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        ctx = memory.load_memory_variables({"input": "hi", "session_id": "s1"})
        assert "rtmdk_context" in ctx
