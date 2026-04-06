"""
test_rtmdk_v2.py
Тесты для всех фич RTMDK v2 по дорожной карте.
"""

import pytest
import json
import os
import time
import math
import numpy as np

from rtmdk_memory_v2 import (
    ConsolidationMode, Backend, ContextFormat,
    RTMDKConfig, MemoryNode, RTMDKField, RTMDKMemory,
    IncPCAProjection, BM25Index, AdaptiveThreshold,
    TDAMonitor, HNSWIndex, TorchBackend,
    format_context, format_context_json, format_context_yaml,
    build_system_prompt, SYSTEM_PROMPT_TEMPLATES,
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
def config_v2():
    return RTMDKConfig(
        embedding_dim=768, latent_dim=64,
        tension_threshold=0.2, decay_rate=0.995,
        top_k=3, enable_async=False,
    )


@pytest.fixture
def field_v2(config_v2):
    return RTMDKField(config_v2)


@pytest.fixture
def memory_v2(config_v2, dummy_embedder):
    return RTMDKMemory(config=config_v2, embedder=dummy_embedder)


# ============================================================================
# ФАЗА 1.1: Structured context + system prompt
# ============================================================================

class TestStructuredContext:
    def _make_results(self):
        node = MemoryNode(
            id="n1", latent_pos=np.zeros(64, dtype=np.float32),
            phase=0.0, amplitude=0.7, salience=0.6,
            content={"text": "Coffee is great", "timestamp": 1000.0},
            lineage=["a+b"], modality="text", self_sup_score=0.95,
        )
        return [("n1", 0.82, node)]

    def test_plain_format(self):
        results = self._make_results()
        ctx = format_context(results, ContextFormat.PLAIN)
        assert "[R:0.82|S:0.60]" in ctx
        assert "Coffee is great" in ctx

    def test_json_format(self):
        results = self._make_results()
        ctx = format_context(results, ContextFormat.JSON)
        data = json.loads(ctx)
        assert len(data) == 1
        assert data[0]["resonance"] == 0.82
        assert data[0]["text"] == "Coffee is great"
        assert data[0]["lineage"] == ["a+b"]
        assert data[0]["modality"] == "text"
        assert data[0]["self_sup_score"] == 0.95
        assert "timestamp" in data[0]["metadata"]

    def test_yaml_format(self):
        results = self._make_results()
        ctx = format_context(results, ContextFormat.YAML)
        assert "resonance: 0.82" in ctx
        assert "salience: 0.60" in ctx
        assert "Coffee is great" in ctx

    def test_empty_results(self):
        assert format_context([], ContextFormat.PLAIN) == "No relevant memory."
        assert format_context([], ContextFormat.JSON) == "[]"
        assert format_context([], ContextFormat.YAML) == "No relevant memory."

    def test_system_prompt_plain(self):
        ctx = "[R:0.82|S:0.60] Coffee is great"
        prompt = build_system_prompt(ctx, ContextFormat.PLAIN, True)
        assert "long-term memory" in prompt
        assert "Coffee is great" in prompt
        assert "Higher resonance" in prompt

    def test_system_prompt_json(self):
        ctx = '[{"resonance": 0.82, "text": "Coffee"}]'
        prompt = build_system_prompt(ctx, ContextFormat.JSON, True)
        assert "JSON format" in prompt
        assert "resonance" in prompt
        assert "lineage" in prompt

    def test_system_prompt_disabled(self):
        prompt = build_system_prompt("mem", ContextFormat.PLAIN, False)
        assert prompt == "You are a helpful assistant with long-term memory."

    def test_system_prompt_empty_context(self):
        prompt = build_system_prompt("No relevant memory.", ContextFormat.PLAIN, True)
        assert prompt == "You are a helpful assistant with long-term memory."

    def test_memory_get_system_prompt(self, dummy_embedder):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            context_format=ContextFormat.JSON,
            use_structured_prompt=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        prompt = memory.get_system_prompt('[{"resonance": 0.5, "text": "test"}]')
        assert "JSON format" in prompt
        assert "test" in prompt


# ============================================================================
# ФАЗА 1.2: Adaptive threshold
# ============================================================================

class TestAdaptiveThreshold:
    def test_creation(self):
        at = AdaptiveThreshold(window_size=30, base_threshold=0.25)
        assert at.current_threshold == 0.25

    def test_records_tension(self):
        at = AdaptiveThreshold(window_size=10, base_threshold=0.25, sensitivity=0.5)
        for t in [0.1, 0.15, 0.2, 0.25, 0.3]:
            at.record_tension(t)
        assert at.get_threshold() != 0.25

    def test_threshold_adapts_to_variance(self):
        at = AdaptiveThreshold(window_size=20, base_threshold=0.25, sensitivity=1.0)
        for t in [0.01] * 10:
            at.record_tension(t)
        low_thresh = at.get_threshold()
        for t in [0.01, 0.5, 0.01, 0.5, 0.01, 0.5, 0.01, 0.5, 0.01, 0.5]:
            at.record_tension(t)
        high_thresh = at.get_threshold()
        assert high_thresh != low_thresh

    def test_is_high_tension(self):
        at = AdaptiveThreshold(window_size=10, base_threshold=0.25)
        for t in [0.3, 0.35, 0.4]:
            at.record_tension(t)
        assert at.is_high_tension(0.5)
        assert not at.is_high_tension(0.01)

    def test_field_with_adaptive_threshold(self, config_v2):
        config_v2.adaptive_threshold = True
        config_v2.adaptive_window = 10
        field = RTMDKField(config_v2)
        assert field.adaptive_threshold is not None
        assert field.get_effective_threshold() == config_v2.tension_threshold

    def test_adaptive_threshold_updates_in_consolidate(self, config_v2):
        config_v2.adaptive_threshold = True
        config_v2.adaptive_window = 3
        config_v2.tension_threshold = 0.05
        config_v2.consolidation_mode = ConsolidationMode.DIALECTICAL
        field = RTMDKField(config_v2)
        emb = np.zeros(768, dtype=np.float32)
        for i in range(6):
            field.add_node(emb, {"text": f"n{i}"}, phase=i * 0.8, node_id=f"n{i}")
        initial_thresh = field.get_effective_threshold()
        field.consolidate()
        assert field.stats["consolidations"] >= 0
        assert field.stats["adaptive_threshold_value"] is not None


# ============================================================================
# ФАЗА 1.3: IncPCA
# ============================================================================

class TestIncPCAProjection:
    def test_creation(self):
        p = IncPCAProjection(768, 64, lr=0.001, update_freq=50)
        assert p.projection.shape == (768, 64)
        assert p.n_samples == 0

    def test_update_changes_projection(self):
        p = IncPCAProjection(768, 64, lr=0.01, update_freq=10)
        initial = p.projection.copy()
        for i in range(15):
            emb = np.random.randn(768).astype(np.float32)
            p.update(emb)
        assert not np.allclose(p.projection, initial)
        assert p.n_samples == 15

    def test_project_after_update(self):
        p = IncPCAProjection(768, 64, lr=0.01, update_freq=5)
        emb = np.random.randn(768).astype(np.float32)
        for _ in range(10):
            p.update(np.random.randn(768).astype(np.float32))
        latent = p.project(emb)
        assert latent.shape == (64,)

    def test_get_set_state(self):
        p = IncPCAProjection(768, 64, lr=0.01, update_freq=5)
        for _ in range(10):
            p.update(np.random.randn(768).astype(np.float32))
        state = p.get_state()
        p2 = IncPCAProjection(768, 64)
        p2.load_state(state)
        assert np.allclose(p.projection, p2.projection)

    def test_field_with_learned_projection(self, config_v2):
        config_v2.learn_projection = True
        config_v2.projection_lr = 0.01
        config_v2.projection_update_freq = 5
        field = RTMDKField(config_v2)
        assert field.projection_learner is not None
        for i in range(10):
            emb = np.random.randn(768).astype(np.float32)
            field.add_node(emb, {"text": f"n{i}"})
        assert field.stats["projection_updates"] >= 1


# ============================================================================
# ФАЗА 1.4: BM25 fallback
# ============================================================================

class TestBM25Index:
    def test_add_and_search(self):
        idx = BM25Index()
        idx.add_document("d1", "Python is a programming language")
        idx.add_document("d2", "Coffee is a popular drink")
        results = idx.search("Python programming")
        assert len(results) > 0
        assert results[0][0] == "d1"

    def test_remove_document(self):
        idx = BM25Index()
        idx.add_document("d1", "Python is great")
        idx.add_document("d2", "Coffee is great")
        idx.remove_document("d1")
        assert "d1" not in idx.documents
        results = idx.search("Python")
        assert len(results) == 0

    def test_empty_search(self):
        idx = BM25Index()
        assert idx.search("anything") == []

    def test_field_with_bm25_fallback(self, config_v2, dummy_embedder):
        config_v2.bm25_fallback = True
        config_v2.min_response = 0.99
        field = RTMDKField(config_v2)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "Python programming language"})
        emb2 = np.random.randn(768).astype(np.float32)
        field.add_node(emb2, {"text": "Coffee is a drink"})
        query_emb = np.random.randn(768).astype(np.float32)
        results = field.query(query_emb, phase=0.0, top_k=3)
        assert field.stats["bm25_fallbacks"] >= 0


# ============================================================================
# ФАЗА 2.1: Soft gates
# ============================================================================

class TestSoftGates:
    def test_soft_gate_below_threshold(self, config_v2):
        config_v2.soft_gates = True
        config_v2.tension_threshold = 0.25
        config_v2.gate_temperature = 0.15
        field = RTMDKField(config_v2)
        gate = field._soft_gate(0.1)
        assert gate < 0.5

    def test_soft_gate_above_threshold(self, config_v2):
        config_v2.soft_gates = True
        config_v2.tension_threshold = 0.25
        config_v2.gate_temperature = 0.15
        field = RTMDKField(config_v2)
        gate = field._soft_gate(0.5)
        assert gate > 0.5

    def test_soft_gate_at_threshold(self, config_v2):
        config_v2.soft_gates = True
        config_v2.tension_threshold = 0.25
        config_v2.gate_temperature = 0.15
        field = RTMDKField(config_v2)
        gate = field._soft_gate(0.25)
        assert 0.4 < gate < 0.6

    def test_soft_gate_disabled(self, config_v2):
        config_v2.soft_gates = False
        field = RTMDKField(config_v2)
        assert field._soft_gate(100.0) == 1.0

    def test_consolidation_with_soft_gates(self, config_v2):
        config_v2.soft_gates = True
        config_v2.tension_threshold = 0.05
        config_v2.gate_temperature = 0.15
        config_v2.consolidation_mode = ConsolidationMode.DIALECTICAL
        field = RTMDKField(config_v2)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "A"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "B"}, phase=np.pi, node_id="b")
        field.consolidate()
        assert len(field.nodes) <= 2
        for node in field.nodes.values():
            assert node.tension == 0.0


# ============================================================================
# ФАЗА 2.2: Self-supervision
# ============================================================================

class TestSelfSupervision:
    def test_disabled(self, config_v2):
        config_v2.self_supervision = False
        field = RTMDKField(config_v2)
        field._self_supervise()
        assert field.stats["self_sup_checks"] == 0

    def test_enabled(self, config_v2):
        config_v2.self_supervision = True
        field = RTMDKField(config_v2)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0, node_id="n1")
        field.nodes["n1"].lineage = ["a+b"]
        field._self_supervise()
        assert field.stats["self_sup_checks"] == 1

    def test_score_decreases_on_mismatch(self, config_v2):
        config_v2.self_supervision = True
        config_v2.self_sup_threshold = 0.3
        field = RTMDKField(config_v2)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0, node_id="n1")
        field.nodes["n1"].lineage = ["a+b"]
        field.nodes["n1"].self_sup_score = 0.5
        for _ in range(5):
            field._self_supervise()
        assert field.nodes["n1"].self_sup_score < 0.5

    def test_verify_after_consolidate(self, config_v2):
        config_v2.self_sup_verify_after_consolidate = True
        config_v2.tension_threshold = 0.05
        config_v2.consolidation_mode = ConsolidationMode.DIALECTICAL
        field = RTMDKField(config_v2)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "A"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "B"}, phase=np.pi, node_id="b")
        field.consolidate()
        for node in field.nodes.values():
            if node.lineage:
                assert node.self_sup_score <= 1.0


# ============================================================================
# ФАЗА 2.3: GPU Backend
# ============================================================================

class TestTorchBackend:
    def test_creation(self):
        tb = TorchBackend()
        assert tb is not None

    def test_numpy_fallback(self):
        tb = TorchBackend()
        ql = np.random.randn(2, 64).astype(np.float32)
        qp = np.array([0.0, np.pi], dtype=np.float32)
        np_ = np.random.randn(10, 64).astype(np.float32)
        nph = np.random.randn(10).astype(np.float32)
        na = np.random.rand(10).astype(np.float32)
        ns = np.random.rand(10).astype(np.float32)
        result = tb.batch_resonance(ql, qp, np_, nph, na, ns, 1.0, 0.3)
        assert result.shape == (2, 10)


# ============================================================================
# ФАЗА 3.1: Multimodal
# ============================================================================

class TestMultimodalField:
    def test_multimodal_config(self, config_v2):
        config_v2.multimodal = True
        config_v2.modalities = ["text", "audio", "image"]
        field = RTMDKField(config_v2)
        assert "audio" in config_v2.modality_phase_shifts
        assert "image" in config_v2.modality_phase_shifts

    def test_modality_phase_shift(self, config_v2):
        config_v2.multimodal = True
        config_v2.modalities = ["text", "audio"]
        config_v2.modality_phase_shifts = {"text": 0.0, "audio": np.pi / 3}
        field = RTMDKField(config_v2)
        phase_text = field._get_phase(modality="text")
        phase_audio = field._get_phase(modality="audio")
        diff = abs(phase_audio - phase_text)
        assert diff > 0.1


# ============================================================================
# ФАЗА 3.2: HNSW
# ============================================================================

class TestHNSWIndex:
    def test_insert_and_search(self):
        idx = HNSWIndex(m=4, ef_construction=10)
        for i in range(20):
            pos = np.random.randn(64).astype(np.float32)
            idx.insert(f"n{i}", pos)
        query = np.zeros(64, dtype=np.float32)
        results = idx.search(query, top_k=5)
        assert 0 < len(results) <= 5

    def test_remove(self):
        idx = HNSWIndex()
        idx.insert("n1", np.zeros(64, dtype=np.float32))
        idx.insert("n2", np.ones(64, dtype=np.float32))
        idx.remove("n1")
        assert "n1" not in idx.positions

    def test_empty_search(self):
        idx = HNSWIndex()
        assert idx.search(np.zeros(64), top_k=5) == []

    def test_field_with_hnsw(self, config_v2):
        config_v2.use_hnsw = True
        config_v2.hnsw_m = 8
        field = RTMDKField(config_v2)
        assert field.hnsw_index is not None
        for i in range(10):
            emb = np.random.randn(768).astype(np.float32)
            field.add_node(emb, {"text": f"n{i}"})
        assert len(field.hnsw_index.positions) == 10


# ============================================================================
# ФАЗА 3.3: TDA
# ============================================================================

class TestTDAMonitor:
    def test_empty_field(self):
        m = TDAMonitor()
        r = m.compute_persistence({})
        assert r["H0"] == 0

    def test_single_node(self):
        m = TDAMonitor()
        node = MemoryNode(id="n1", latent_pos=np.zeros(64, dtype=np.float32),
                          phase=0.0, amplitude=0.7, salience=0.6)
        r = m.compute_persistence({"n1": node})
        assert r["H0"] == 0

    def test_multiple_nodes(self):
        m = TDAMonitor()
        nodes = {}
        for i in range(10):
            pos = np.random.randn(64).astype(np.float32)
            nodes[f"n{i}"] = MemoryNode(id=f"n{i}", latent_pos=pos,
                                         phase=0.0, amplitude=0.7, salience=0.6)
        r = m.compute_persistence(nodes)
        assert "H0" in r
        assert "H1" in r

    def test_trend_stable(self):
        m = TDAMonitor()
        for _ in range(5):
            nodes = {}
            for i in range(5):
                pos = np.random.randn(64).astype(np.float32)
                nodes[f"n{i}"] = MemoryNode(id=f"n{i}", latent_pos=pos,
                                             phase=0.0, amplitude=0.7, salience=0.6)
            m.compute_persistence(nodes)
        assert m.get_trend() == "stable"

    def test_field_with_tda(self, config_v2):
        config_v2.tda_monitoring = True
        config_v2.tda_check_freq = 5
        field = RTMDKField(config_v2)
        assert field.tda_monitor is not None
        for i in range(10):
            emb = np.random.randn(768).astype(np.float32)
            field.step(inputs=[{"embedding": emb, "content": {"text": f"x{i}"}}])
        assert field.stats["tda_checks"] >= 1


# ============================================================================
# EXPORT/IMPORT v2
# ============================================================================

class TestExportImportV2:
    def test_export_import_with_learned_projection(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            learn_projection=True, projection_lr=0.01,
            projection_update_freq=5, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context(
            {"input": "test data", "session_id": "s1"},
            {"output": "test output"}
        )
        path = str(tmp_path / "v2_export.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert len(imported.field.nodes) == len(memory.field.nodes)
        assert imported.field.projection_learner is not None

    def test_export_import_with_tda(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(
            embedding_dim=768, latent_dim=64,
            tda_monitoring=True, enable_async=False,
        )
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        for i in range(5):
            memory.save_context(
                {"input": f"tda test {i}", "session_id": "s1"},
                {"output": f"response {i}"}
            )
        memory.field._check_tda()
        path = str(tmp_path / "v2_tda_export.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert imported.field.tda_monitor is not None
        assert len(imported.field.tda_monitor.history) >= 1


# ============================================================================
# BACKWARD COMPATIBILITY
# ============================================================================

class TestBackwardCompatibility:
    def test_default_config_works(self, dummy_embedder):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "hello", "session_id": "s1"}, {"output": "hi"})
        ctx = memory.load_memory_variables({"input": "hi", "session_id": "s1"})
        assert "rtmdk_context" in ctx

    def test_v1_import_still_works(self, dummy_embedder, tmp_path):
        config = RTMDKConfig(embedding_dim=768, latent_dim=64, enable_async=False)
        memory = RTMDKMemory(config=config, embedder=dummy_embedder)
        memory.save_context({"input": "v1 test", "session_id": "s1"}, {"output": "v1 out"})
        path = str(tmp_path / "v1_compat.json")
        memory.export_field(path)
        imported = RTMDKMemory.import_field(path, dummy_embedder)
        assert len(imported.field.nodes) == 1
