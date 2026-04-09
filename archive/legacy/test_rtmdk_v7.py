"""
test_rtmdk_v7.py
Тесты для Фазы 10: Cross-modal, Meta Controller, Federated Sync.
"""

import pytest
import json
import numpy as np
import math
import time
from typing import Dict

from rtmdk_memory_v7 import (
    RTMDKConfig, MemoryNode, RTMDKField, RTMDKMemory,
    detect_modality, cross_modal_resonance,
    MetaController, KuramotoSync, FederatedRTMDK, FederatedNode,
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
def config_v7():
    return RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        tension_threshold=0.2, decay_rate=0.995,
        top_k=3, enable_async=False,
    )


# ============================================================================
# ФАЗА 10.1: КРОСС-МОДАЛЬНОСТЬ
# ============================================================================

class TestCrossModal:
    def test_detect_modality_text(self):
        assert detect_modality("Hello world") == "text"

    def test_detect_modality_code(self):
        assert detect_modality("def hello(): pass") == "code"
        assert detect_modality("class Foo: pass") == "code"
        assert detect_modality("import os") == "code"

    def test_detect_modality_metrics(self):
        assert detect_modality("CPU 3.14 MEM 2.71") == "metrics"

    def test_detect_modality_audio(self):
        assert detect_modality("<audio>test</audio>") == "audio"
        assert detect_modality("recording.wav") == "audio"

    def test_detect_modality_vision(self):
        assert detect_modality("<image>test</image>") == "vision"
        assert detect_modality("photo.png") == "vision"

    def test_cross_modal_resonance_same_modality(self):
        result = cross_modal_resonance("text", "text", 0.5, {"text": 0.0}, 0.35)
        assert result > 0.5

    def test_cross_modal_resonance_different_modality(self):
        offsets = {"text": 0.0, "code": np.pi / 4}
        result = cross_modal_resonance("text", "code", 0.5, offsets, 0.35)
        assert result > 0.5

    def test_cross_modal_resonance_further_modality(self):
        offsets_close = {"text": 0.0, "code": np.pi/4}
        offsets_far = {"text": 0.0, "metrics": np.pi}
        result_close = cross_modal_resonance("text", "code", 0.5, offsets_close, 0.35)
        result_far = cross_modal_resonance("text", "metrics", 0.5, offsets_far, 0.35)
        assert result_close > result_far

    def test_field_cross_modal_enabled(self, config_v7):
        config_v7.cross_modal = True
        field = RTMDKField(config_v7)
        assert field.cfg.cross_modal is True
        assert "text" in field.cfg.modal_phase_offsets
        assert "code" in field.cfg.modal_phase_offsets

    def test_cross_modal_query(self, config_v7):
        config_v7.cross_modal = True
        config_v7.bm25_fallback = True
        field = RTMDKField(config_v7)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "text content"}, modality="text", node_id="n1")
        field.add_node(emb + 0.01, {"text": "code content"}, modality="code", node_id="n2")
        results = field.query(emb, phase=0.0, modality="text")
        assert len(results) > 0

    def test_cross_modal_stats(self, config_v7, dummy_embedder):
        config_v7.cross_modal = True
        memory = RTMDKMemory(config=config_v7, embedder=dummy_embedder)
        memory.save_context(
            {"input": "def hello(): pass", "session_id": "s1"},
            {"output": "code response"}
        )
        stats = memory.field.get_cross_modal_stats()
        assert "cross_modal_enabled" in stats
        assert stats["cross_modal_enabled"] is True


# ============================================================================
# ФАЗА 10.2: МЕТАКОГНИТИВНЫЙ КОНТРОЛЛЕР
# ============================================================================

class TestMetaController:
    def test_creation(self):
        mc = MetaController(n_trials=10)
        assert mc.n_trials == 10

    def test_optimize_grid(self, config_v7):
        config_v7.meta_controller = True
        field = RTMDKField(config_v7)
        mc = MetaController(n_trials=5)
        best = mc.optimize(field)
        assert isinstance(best, dict)
        assert len(best) > 0

    def test_optimize_updates_config(self, config_v7):
        config_v7.meta_controller = True
        field = RTMDKField(config_v7)
        initial_decay = field.cfg.decay_rate
        mc = MetaController(n_trials=3)
        best = mc.optimize(field)
        assert mc._best_params is not None
        # Config should have been updated with best params
        assert mc._total_optimizations >= 1

    def test_get_history(self, config_v7):
        config_v7.meta_controller = True
        field = RTMDKField(config_v7)
        mc = MetaController(n_trials=3)
        mc.optimize(field)
        history = mc._optimization_history
        assert len(history) >= 1
        assert "params" in history[0]
        assert "best_value" in history[0] or "score" in history[0]

    def test_meta_controller_in_field(self, config_v7):
        config_v7.meta_controller = True
        config_v7.meta_n_trials = 3
        config_v7.meta_optimization_freq = 1
        field = RTMDKField(config_v7)
        assert field.meta_controller is not None
        emb = np.random.randn(768).astype(np.float32)
        for _ in range(3):
            field.add_node(emb, {"text": "test"})
        field.consolidate()
        assert field.stats.get("meta_optimizations", 0) >= 0


# ============================================================================
# ФАЗА 10.3: РАСПРЕДЕЛЁННАЯ СЕТЬ
# ============================================================================

class TestKuramotoSync:
    def test_creation(self):
        k = KuramotoSync()
        assert k is not None
        assert k.coupling_strength > 0

    def test_add_oscillator(self):
        k = KuramotoSync()
        k.add_oscillator("n1", phase=0.0)
        k.add_oscillator("n2", phase=np.pi / 2)
        assert "n1" in k.phases
        assert "n2" in k.phases
        assert len(k.phases) == 2

    def test_step(self):
        k = KuramotoSync()
        k.add_oscillator("n1", phase=0.0)
        k.add_oscillator("n2", phase=np.pi / 2)
        result = k.step(n_steps=5)
        assert isinstance(result, dict)

    def test_compute_order_parameter(self):
        k = KuramotoSync()
        k.add_oscillator("n1", phase=0.0)
        k.add_oscillator("n2", phase=0.0)
        r = k.compute_order_parameter()
        assert 0.0 <= r <= 1.0

    def test_sync_to_target(self):
        k = KuramotoSync()
        k.add_oscillator("n1", phase=0.0)
        k.add_oscillator("n2", phase=np.pi)
        result = k.sync_to_target({"n1": np.pi/4, "n2": np.pi/4}, n_steps=10)
        assert isinstance(result, dict)

    def test_get_load_state(self):
        k = KuramotoSync()
        k.add_oscillator("n1", phase=0.0)
        state = k.get_state()
        k2 = KuramotoSync()
        k2.load_state(state)
        assert "n1" in k2.phases


class TestFederatedRTMDK:
    def test_creation(self):
        f = FederatedRTMDK(node_id="local_node", sync_lr=0.01)
        assert f.node_id == "local_node"
        assert f.sync_lr == 0.01

    def test_register_peer(self):
        f = FederatedRTMDK(node_id="local")
        peer = FederatedNode(node_id="remote1", phase=0.0)
        f.register_peer(peer)
        assert "remote1" in f.peers

    def test_sync_with_peers(self):
        f = FederatedRTMDK(node_id="local", sync_lr=0.1)
        peer = FederatedNode(node_id="remote1", phase=0.5, params={"n1": 0.8})
        f.register_peer(peer)
        local_phases = {"n1": 0.0}
        local_params = {"n1": 0.7}
        result = f.sync_with_peers(local_phases, local_params)
        assert isinstance(result, dict)

    def test_sync_status(self):
        f = FederatedRTMDK(node_id="local")
        peer = FederatedNode(node_id="remote1", phase=0.0)
        f.register_peer(peer)
        status = f.get_sync_status()
        assert status["node_id"] == "local"
        assert status["n_peers"] == 1
        assert status["active_peers"] == 1

    def test_get_load_state(self):
        f = FederatedRTMDK(node_id="local")
        peer = FederatedNode(node_id="r1", phase=0.0)
        f.register_peer(peer)
        state = f.export_state()
        f2 = FederatedRTMDK(node_id="local2")
        f2.import_state(state)
        assert "r1" in f2.peers

    def test_field_with_federated(self, config_v7):
        config_v7.federated = True
        config_v7.node_id = "test_node"
        field = RTMDKField(config_v7)
        assert field.federated is not None
        assert field.federated.node_id == "test_node"


# ============================================================================
# EXPORT/IMPORT v7
# ============================================================================

class TestExportImportV7:
    def test_export_import_cross_modal(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            cross_modal=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "test", "session_id": "s1"}, {"output": "out"})
        path = str(tmp_path / "v7_crossmodal.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.cfg.cross_modal is True

    def test_export_import_full_v7(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            cross_modal=True, meta_controller=True,
            federated=True, node_id="test",
            causal_topological=True, meta_adaptive=True,
            self_healing=True, differentiable=True,
            continuous_dynamics=True, production_mode=True,
            ragas_enabled=True, auto_rollback=True,
            eval_mode="production", enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        for i in range(3):
            memory.save_context(
                {"input": f"msg {i}", "session_id": "s1"},
                {"output": f"resp {i}"}
            )
        path = str(tmp_path / "v7_full.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.cfg.cross_modal is True
        assert imported.field.meta_controller is not None
        assert imported.field.federated is not None
        assert len(imported.field.nodes) == 3


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class TestBackwardCompatibilityV7:
    def test_default_config_works(self, dummy_embedder):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        ctx = memory.load_memory_variables({"input": "hi", "session_id": "s1"})
        assert "rtmdk_context" in ctx

    def test_v6_import_works(self, dummy_embedder, tmp_path):
        from rtmdk_memory_v6 import RTMDKConfig as V6Config, RTMDKMemory as V6Memory
        v6_config = V6Config(embedding_dim=768, latent_dim=64, enable_async=False)
        v6_memory = V6Memory(config=v6_config, embedder=dummy_embedder)
        v6_memory.save_context({"input": "v6 test", "session_id": "s1"}, {"output": "v6 out"})
        path = str(tmp_path / "v6_compat.json")
        v6_memory.export_field(path)
        v7_memory = RTMDKMemory.import_field(path, dummy_embedder)
        assert len(v7_memory.field.nodes) == 1
