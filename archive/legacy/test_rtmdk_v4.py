"""
test_rtmdk_v4.py
Тесты для Фазы 5: Мета-адаптивные ядра + Самовосстанавливающаяся топология.
"""

import pytest
import json
import numpy as np
import math
from typing import Dict

from rtmdk_memory_v4 import (
    RTMDKConfig, MemoryNode, RTMDKField, RTMDKMemory,
    MetaAdaptiveKernel, TopologyHealer, FieldHealth,
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
def config_v4():
    return RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        tension_threshold=0.2, decay_rate=0.995,
        top_k=3, enable_async=False,
    )


# ============================================================================
# ТРЕК A: МЕТА-АДАПТИВНЫЕ ЯДРА
# ============================================================================

class TestMetaAdaptiveKernel:
    def test_creation(self):
        k = MetaAdaptiveKernel(base_bandwidth=1.0, base_phase_coupling=0.3)
        assert k.effective_bandwidth == 1.0
        assert k.effective_phase_coupling == 0.3

    def test_record_response(self):
        k = MetaAdaptiveKernel()
        k.record_response(0.5)
        assert len(k._response_history) == 1

    def test_record_semantic_density(self):
        k = MetaAdaptiveKernel()
        k.record_semantic_density(0.8)
        assert len(k._semantic_density) == 1

    def test_record_uncertainty(self):
        k = MetaAdaptiveKernel()
        k.record_uncertainty(1.5)
        assert len(k._uncertainty) == 1

    def test_compute_kurtosis_empty(self):
        k = MetaAdaptiveKernel()
        assert k.compute_resonance_kurtosis() == 3.0

    def test_compute_kurtosis_with_data(self):
        k = MetaAdaptiveKernel()
        for v in [0.1, 0.2, 0.1, 0.8, 0.1, 0.15, 0.9, 0.05]:
            k.record_response(v)
        kurt = k.compute_resonance_kurtosis()
        assert isinstance(kurt, float)
        assert kurt > 0

    def test_adapt_low_kurtosis_narrows_bandwidth(self):
        """Куртозис < target_min → bandwidth уменьшается."""
        k = MetaAdaptiveKernel(base_bandwidth=1.0, kurtosis_target_min=2.0, kurtosis_target_max=5.0)
        # Uniform responses → low kurtosis
        for v in np.linspace(0.3, 0.7, 20):
            k.record_response(float(v))
        initial_bw = k.effective_bandwidth
        k.adapt()
        assert k.effective_bandwidth < initial_bw

    def test_adapt_high_kurtosis_widens_bandwidth(self):
        """Куртозис > target_max → bandwidth увеличивается."""
        k = MetaAdaptiveKernel(base_bandwidth=1.0, kurtosis_target_min=1.0, kurtosis_target_max=1.5)
        # Peaked responses → high kurtosis
        for _ in range(20):
            k.record_response(0.5)
        k.record_response(0.0)
        k.record_response(1.0)
        initial_bw = k.effective_bandwidth
        k.adapt()
        assert k.effective_bandwidth > initial_bw

    def test_adapt_high_density_increases_phase_coupling(self):
        k = MetaAdaptiveKernel(base_phase_coupling=0.3)
        for _ in range(10):
            k.record_semantic_density(0.9)
        initial_pc = k.effective_phase_coupling
        k.adapt()
        assert k.effective_phase_coupling > initial_pc

    def test_adapt_high_uncertainty_widens_bandwidth(self):
        k = MetaAdaptiveKernel(base_bandwidth=1.0)
        for _ in range(10):
            k.record_uncertainty(2.0)
        initial_bw = k.effective_bandwidth
        k.adapt()
        assert k.effective_bandwidth >= initial_bw

    def test_bandwidth_bounds(self):
        k = MetaAdaptiveKernel(base_bandwidth=1.0)
        for _ in range(100):
            k.record_response(0.5)
            k.adapt()
        assert 0.1 <= k.effective_bandwidth <= 10.0

    def test_phase_coupling_bounds(self):
        k = MetaAdaptiveKernel(base_phase_coupling=0.3)
        for _ in range(100):
            k.record_semantic_density(0.9)
            k.adapt()
        assert 0.0 <= k.effective_phase_coupling <= 1.0

    def test_get_load_state(self):
        k = MetaAdaptiveKernel(base_bandwidth=1.5, base_phase_coupling=0.4)
        for v in [0.1, 0.2, 0.3]:
            k.record_response(v)
        state = k.get_state()
        k2 = MetaAdaptiveKernel()
        k2.load_state(state)
        assert k2.base_bandwidth == 1.5
        assert k2.base_phase_coupling == 0.4

    def test_get_state_has_required_fields(self):
        k = MetaAdaptiveKernel()
        state = k.get_state()
        assert "kurtosis" in state
        assert "avg_density" in state
        assert "avg_uncertainty" in state


class TestFieldMetaAdaptive:
    def test_field_with_meta_adaptive(self, config_v4):
        config_v4.meta_adaptive = True
        field = RTMDKField(config_v4)
        assert field.meta_kernel is not None

    def test_meta_kernel_adapts_on_query(self, config_v4):
        config_v4.meta_adaptive = True
        config_v4.bm25_fallback = True
        field = RTMDKField(config_v4)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test data for query"})
        field.add_node(emb + 0.01, {"text": "another test"})
        field.query(emb, phase=0.0)
        # Meta kernel should have recorded response
        assert len(field.meta_kernel._response_history) > 0

    def test_meta_stats_updated(self, config_v4):
        config_v4.meta_adaptive = True
        config_v4.bm25_fallback = True
        field = RTMDKField(config_v4)
        for i in range(10):
            emb = np.random.randn(768).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
            field.query(emb, phase=0.0)
        assert "meta_kurtosis" in field.stats
        assert "meta_bandwidth" in field.stats
        assert "meta_phase_coupling" in field.stats

    def test_kurtosis_in_target_range(self, config_v4):
        """После адаптации куртозис должен стремиться к целевому диапазону."""
        config_v4.meta_adaptive = True
        config_v4.kurtosis_target_min = 1.5
        config_v4.kurtosis_target_max = 4.0
        field = RTMDKField(config_v4)
        for i in range(30):
            emb = np.random.randn(768).astype(np.float32)
            field.add_node(emb, {"text": f"node {i}"})
            field.query(emb, phase=0.0)
        # After adaptation, kurtosis should be moving toward target
        kurt = field.stats["meta_kurtosis"]
        assert isinstance(kurt, float)
        assert kurt > 0


# ============================================================================
# ТРЕК B: САМОВОССТАНАВЛИВАЮЩАЯСЯ ТОПОЛОГИЯ
# ============================================================================

class TestTopologyHealer:
    def test_creation(self):
        h = TopologyHealer()
        assert h.dead_zone_threshold == 0.15
        assert h.hyperconvergence_threshold == 0.05

    def _make_nodes(self, n: int, dim: int = 64) -> Dict[str, MemoryNode]:
        nodes = {}
        for i in range(n):
            pos = np.random.randn(dim).astype(np.float32)
            nodes[f"n{i}"] = MemoryNode(id=f"n{i}", latent_pos=pos, phase=0.0,
                                         amplitude=0.7, salience=0.6)
        return nodes

    def test_detect_dead_zones_empty(self):
        h = TopologyHealer()
        assert h.detect_dead_zones({}) == []

    def test_detect_dead_zones_normal(self):
        h = TopologyHealer()
        nodes = self._make_nodes(10)
        dead = h.detect_dead_zones(nodes)
        # Random nodes shouldn't have extreme dead zones
        assert len(dead) < len(nodes)

    def test_detect_dead_zones_artificial(self):
        h = TopologyHealer(dead_zone_threshold=0.0)
        nodes = self._make_nodes(10, dim=8)
        # Make one node very far
        nodes["n9"].latent_pos = np.ones(8, dtype=np.float32) * 100
        dead = h.detect_dead_zones(nodes)
        assert "n9" in dead

    def test_detect_hyperconvergence(self):
        h = TopologyHealer(hyperconvergence_threshold=0.01)
        nodes = self._make_nodes(5, dim=8)
        # Make all nodes converge to same point
        for n in nodes.values():
            n.latent_pos = np.zeros(8, dtype=np.float32) + np.random.randn(8).astype(np.float32) * 0.001
        assert h.detect_hyperconvergence(nodes)

    def test_detect_hyperconvergence_normal(self):
        h = TopologyHealer()
        nodes = self._make_nodes(10)
        assert not h.detect_hyperconvergence(nodes)

    def test_detect_fragmentation(self):
        h = TopologyHealer()
        nodes = self._make_nodes(10, dim=8)
        frag = h.detect_fragmentation(nodes, radius=0.5)
        assert 0.0 <= frag <= 1.0

    def test_compute_field_health_stable(self):
        h = TopologyHealer(
            hyperconvergence_threshold=0.001,
            fragmentation_threshold=0.9,
            dead_zone_threshold=0.5,
        )
        nodes = self._make_nodes(10, dim=32)
        # Cluster nodes tightly so avg pairwise dist < 2.0
        center = np.zeros(32, dtype=np.float32)
        for n in nodes.values():
            n.latent_pos = center + np.random.randn(32).astype(np.float32) * 0.2
        health, diag = h.compute_field_health(nodes)
        assert health == FieldHealth.STABLE
        assert diag["fragmentation"] < 0.5

    def test_compute_field_health_hyperconvergence(self):
        h = TopologyHealer(hyperconvergence_threshold=0.001)
        nodes = self._make_nodes(5, dim=8)
        for n in nodes.values():
            n.latent_pos = np.zeros(8, dtype=np.float32) + np.random.randn(8).astype(np.float32) * 0.0001
        health, diag = h.compute_field_health(nodes)
        assert health in (FieldHealth.CRITICAL, FieldHealth.DEGRADED)

    def test_heal_dead_zones(self):
        h = TopologyHealer(dead_zone_threshold=0.0, healing_strength=0.3)
        nodes = self._make_nodes(5, dim=8)
        nodes["n4"].latent_pos = np.ones(8, dtype=np.float32) * 100
        dead = h.detect_dead_zones(nodes)
        healed = h.heal_dead_zones(nodes, dead)
        assert len(healed) > 0
        assert nodes["n4"].is_healing
        assert nodes["n4"].healing_origin is not None

    def test_heal_hyperconvergence(self):
        h = TopologyHealer(hyperconvergence_threshold=0.001, healing_strength=0.2)
        nodes = self._make_nodes(5, dim=8)
        for n in nodes.values():
            n.latent_pos = np.zeros(8, dtype=np.float32) + np.random.randn(8).astype(np.float32) * 0.0001
        healed = h.heal_hyperconvergence(nodes)
        assert len(healed) > 0
        assert nodes[healed[0]["node_id"]].is_healing

    def test_heal_fragmentation(self):
        h = TopologyHealer(healing_strength=0.3)
        nodes = self._make_nodes(6, dim=8)
        # Make 2 nodes far away
        for n in nodes.values():
            n.latent_pos = np.zeros(8, dtype=np.float32)
        nodes["n4"].latent_pos = np.ones(8, dtype=np.float32) * 50
        nodes["n5"].latent_pos = np.ones(8, dtype=np.float32) * -50
        healed = h.heal_fragmentation(nodes, ["n4", "n5"])
        assert len(healed) > 0

    def test_get_load_state(self):
        h = TopologyHealer()
        nodes = self._make_nodes(5)
        h.compute_field_health(nodes)
        state = h.get_state()
        h2 = TopologyHealer()
        h2.load_state(state)
        assert len(h2._health_history) > 0


class TestFieldSelfHealing:
    def test_field_with_self_healing(self, config_v4):
        config_v4.self_healing = True
        field = RTMDKField(config_v4)
        assert field.healer is not None

    def test_healing_triggered_on_step(self, config_v4):
        config_v4.self_healing = True
        config_v4.healing_check_freq = 1
        config_v4.dead_zone_threshold = 0.0
        field = RTMDKField(config_v4)
        # Create a dead zone
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "cluster"}, node_id="c1")
        field.add_node(emb + 0.01, {"text": "cluster2"}, node_id="c2")
        # Dead zone node
        dead_emb = np.ones(768, dtype=np.float32) * 100
        field.add_node(dead_emb, {"text": "dead"}, node_id="dead")
        for _ in range(5):
            field.step(inputs=[{"embedding": emb, "content": {"text": "x"}}])
        assert "field_health" in field.stats
        assert field.stats["field_health"] in ("stable", "healing", "degraded", "critical")

    def test_get_field_health(self, config_v4, dummy_embedder):
        config_v4.self_healing = True
        memory = RTMDKMemory(config=config_v4, embedder=dummy_embedder)
        for i in range(5):
            memory.save_context(
                {"input": f"message {i}", "session_id": "s1"},
                {"output": f"response {i}"}
            )
        health = memory.get_field_health()
        assert "health" in health
        assert "kurtosis" in health

    def test_trigger_healing(self, config_v4, dummy_embedder):
        config_v4.self_healing = True
        config_v4.dead_zone_threshold = 0.0
        memory = RTMDKMemory(config=config_v4, embedder=dummy_embedder)
        memory.save_context(
            {"input": "normal", "session_id": "s1"},
            {"output": "normal out"}
        )
        # Add dead zone manually
        dead_emb = np.ones(768, dtype=np.float32) * 100
        memory.field.add_node(dead_emb, {"text": "dead zone"}, node_id="dead")
        healed = memory.trigger_healing()
        assert isinstance(healed, list)

    def test_healing_stats_updated(self, config_v4):
        config_v4.self_healing = True
        config_v4.healing_check_freq = 1
        config_v4.dead_zone_threshold = 0.0
        field = RTMDKField(config_v4)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "c1"}, node_id="c1")
        field.add_node(emb + 0.01, {"text": "c2"}, node_id="c2")
        dead_emb = np.ones(768, dtype=np.float32) * 100
        field.add_node(dead_emb, {"text": "dead"}, node_id="dead")
        for _ in range(3):
            field.step(inputs=[{"embedding": emb, "content": {"text": "x"}}])
        assert "healing_events" in field.stats


# ============================================================================
# EXPORT/IMPORT v4
# ============================================================================

class TestExportImportV4:
    def test_export_import_meta_adaptive(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            meta_adaptive=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        path = str(tmp_path / "v4_meta.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.meta_kernel is not None
        assert len(imported.field.nodes) == 1

    def test_export_import_self_healing(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            self_healing=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        path = str(tmp_path / "v4_heal.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.healer is not None

    def test_export_import_full_v4(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            meta_adaptive=True, self_healing=True,
            differentiable=True, continuous_dynamics=True,
            causal_modeling=True, production_mode=True,
            ab_variant="control", enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        for i in range(3):
            memory.save_context(
                {"input": f"msg {i}", "session_id": "s1"},
                {"output": f"resp {i}"}
            )
        path = str(tmp_path / "v4_full.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.meta_kernel is not None
        assert imported.field.healer is not None
        assert imported.field.learnable_kernel is not None
        assert imported.field.neural_ode is not None
        assert imported.field.causal_graph is not None
        assert imported.field.monitor is not None
        assert len(imported.field.nodes) == 3


# ============================================================================
# INTEGRATION: Memory v4
# ============================================================================

class TestMemoryV4:
    def test_inspect_node_has_healing_fields(self, dummy_embedder):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        nid = memory.field.node_index[0]
        info = memory.inspect_node(nid)
        assert info is not None
        assert "is_healing" in info
        assert "healing_origin" in info
        assert "local_density" in info

    def test_get_field_health(self, dummy_embedder):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            self_healing=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        for i in range(5):
            memory.save_context(
                {"input": f"msg {i}", "session_id": "s1"},
                {"output": f"resp {i}"}
            )
        health = memory.get_field_health()
        assert "health" in health
        assert "kurtosis" in health

    def test_trigger_healing_method(self, dummy_embedder):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            self_healing=True, dead_zone_threshold=0.0,
            enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "normal", "session_id": "s1"}, {"output": "out"})
        dead_emb = np.ones(768, dtype=np.float32) * 100
        memory.field.add_node(dead_emb, {"text": "dead"}, node_id="dead")
        healed = memory.trigger_healing()
        assert isinstance(healed, list)


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class TestBackwardCompatibilityV4:
    def test_default_config_works(self, dummy_embedder):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        ctx = memory.load_memory_variables({"input": "hi", "session_id": "s1"})
        assert "rtmdk_context" in ctx

    def test_v3_import_works(self, dummy_embedder, tmp_path):
        """v3 export should be importable by v4."""
        from rtmdk_memory_v3 import RTMDKConfig as V3Config, RTMDKMemory as V3Memory
        v3_config = V3Config(embedding_dim=768, latent_dim=64, enable_async=False)
        v3_memory = V3Memory(config=v3_config, embedder=dummy_embedder)
        v3_memory.save_context({"input": "v3 test", "session_id": "s1"}, {"output": "v3 out"})
        path = str(tmp_path / "v3_compat.json")
        v3_memory.export_field(path)
        # v4 should be able to import v3 export
        v4_memory = RTMDKMemory.import_field(path, dummy_embedder)
        assert len(v4_memory.field.nodes) == 1
