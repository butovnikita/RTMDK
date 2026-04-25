"""
Тесты для RTMDK Memory системы.
Покрывает RTMDKConfig, MemoryNode, RTMDKField и RTMDKMemory.
"""

import pytest
import json
import os
import time
import numpy as np
from unittest.mock import patch, MagicMock

from rtmdk_memory import (
    ConsolidationMode,
    RTMDKConfig,
    MemoryNode,
    RTMDKField,
    RTMDKMemory,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def dummy_embedder():
    """Простой детерминированный эмбеддер для тестов."""
    def _embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base
    return _embed


@pytest.fixture
def config():
    return RTMDKConfig(
        embedding_dim=768,
        latent_dim=64,
        tension_threshold=0.2,
        decay_rate=0.995,
        top_k=3,
        enable_async=False,
    )


@pytest.fixture
def field(config):
    return RTMDKField(config)


@pytest.fixture
def memory(config, dummy_embedder):
    return RTMDKMemory(config=config, embedder=dummy_embedder)


# ============================================================================
# RTMDKConfig TESTS
# ============================================================================

class TestRTMDKConfig:
    def test_default_values(self):
        cfg = RTMDKConfig()
        assert cfg.embedding_dim == 768
        assert cfg.latent_dim == 64
        assert cfg.resonance_kernel == "gaussian_phase"
        assert cfg.phase_coupling == 0.3
        assert cfg.bandwidth == 1.0
        assert cfg.attraction_lr == 0.02
        assert cfg.phase_sync_lr == 0.01
        assert cfg.decay_rate == 0.998
        assert cfg.min_amplitude == 0.05
        assert cfg.tension_threshold == 0.25
        assert cfg.consolidation_mode == ConsolidationMode.DIALECTICAL
        assert cfg.max_nodes == 5000
        assert cfg.top_k == 5
        assert cfg.min_response == 0.1
        assert cfg.enable_async is True

    def test_custom_values(self):
        cfg = RTMDKConfig(
            embedding_dim=384,
            latent_dim=32,
            decay_rate=0.99,
            top_k=10,
        )
        assert cfg.embedding_dim == 384
        assert cfg.latent_dim == 32
        assert cfg.decay_rate == 0.99
        assert cfg.top_k == 10


# ============================================================================
# MemoryNode TESTS
# ============================================================================

class TestMemoryNode:
    def test_creation(self):
        node = MemoryNode(
            id="test_1",
            latent_pos=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            phase=1.57,
            amplitude=0.8,
            salience=0.6,
            content={"text": "hello"},
        )
        assert node.id == "test_1"
        assert node.phase == 1.57
        assert node.amplitude == 0.8
        assert node.salience == 0.6
        assert node.tension == 0.0
        assert node.content == {"text": "hello"}
        assert node.lineage == []

    def test_to_dict(self):
        node = MemoryNode(
            id="n1",
            latent_pos=np.array([1.0, 2.0], dtype=np.float32),
            phase=0.5,
            amplitude=0.7,
            salience=0.4,
            content={"text": "test"},
        )
        d = node.to_dict()
        assert d["id"] == "n1"
        assert isinstance(d["latent_pos"], list)
        assert d["latent_pos"] == [1.0, 2.0]
        assert d["content"]["text"] == "test"

    def test_from_dict(self):
        data = {
            "id": "n2",
            "latent_pos": [0.5, 1.5],
            "phase": 2.0,
            "amplitude": 0.9,
            "salience": 0.3,
            "tension": 0.1,
            "content": {"text": "restored"},
            "created_at": 1000.0,
            "last_resonated": 500.0,
            "lineage": ["a+b"],
        }
        node = MemoryNode.from_dict(data)
        assert node.id == "n2"
        assert isinstance(node.latent_pos, np.ndarray)
        assert node.latent_pos.dtype == np.float32
        assert node.content == {"text": "restored"}
        assert node.lineage == ["a+b"]


# ============================================================================
# RTMDKField TESTS
# ============================================================================

class TestRTMDKField:
    def test_initialization(self, field, config):
        assert field.cfg is config
        assert len(field.nodes) == 0
        assert len(field.node_index) == 0
        assert field.projection.shape == (768, 64)
        assert field.stats["total_adds"] == 0

    def test_projection(self, field):
        emb = np.random.randn(768).astype(np.float32)
        latent = field._project(emb)
        assert latent.shape == (64,)
        assert latent.dtype == np.float32

    def test_add_node(self, field):
        emb = np.random.randn(768).astype(np.float32)
        nid = field.add_node(emb, {"text": "test"})
        assert nid in field.nodes
        assert nid in field.node_index
        assert field.nodes[nid].content["text"] == "test"
        assert field.stats["total_adds"] == 1

    def test_add_node_custom_id(self, field):
        emb = np.random.randn(768).astype(np.float32)
        nid = field.add_node(emb, {"text": "test"}, phase=1.0, node_id="custom_id")
        assert nid == "custom_id"
        assert field.nodes["custom_id"].phase == 1.0

    def test_query_empty_field(self, field):
        emb = np.random.randn(768).astype(np.float32)
        results = field.query(emb)
        assert results == []

    def test_query_with_nodes(self, field):
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0)
        results = field.query(emb, phase=0.0)
        assert len(results) >= 1
        assert isinstance(results[0], tuple)
        assert len(results[0]) == 3  # (nid, response, node)

    def test_query_returns_sorted_results(self, field):
        emb1 = np.random.randn(768).astype(np.float32)
        emb2 = np.random.randn(768).astype(np.float32)
        field.add_node(emb1, {"text": "node1"}, phase=0.0)
        field.add_node(emb2, {"text": "node2"}, phase=np.pi)
        results = field.query(emb1, phase=0.0, top_k=5)
        if len(results) >= 2:
            assert results[0][1] >= results[1][1]

    def test_resonance_response_gaussian(self, config):
        config.resonance_kernel = "gaussian"
        field = RTMDKField(config)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0)
        nid = field.node_index[0]
        node = field.nodes[nid]
        latent = field._project(emb)
        resp = field._resonance_response(latent, 0.0, node)
        assert resp > 0

    def test_resonance_response_cosine(self, config):
        config.resonance_kernel = "cosine"
        field = RTMDKField(config)
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0)
        nid = field.node_index[0]
        node = field.nodes[nid]
        latent = field._project(emb)
        resp = field._resonance_response(latent, 0.0, node)
        assert resp >= 0

    def test_compute_tension_single_node(self, field):
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "solo"})
        nid = field.node_index[0]
        tension = field._compute_tension(nid)
        assert tension == 0.0  # нет соседей

    def test_compute_tension_low_variance(self, field):
        """Узлы с одинаковой фазой и значимостью = низкое напряжение."""
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "a"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "b"}, phase=0.0, node_id="b")
        field.add_node(emb, {"text": "c"}, phase=0.0, node_id="c")
        tension = field._compute_tension("a")
        assert tension < 0.1

    def test_consolidate_no_tension(self, field):
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "a"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "b"}, phase=0.0, node_id="b")
        updated = field.consolidate()
        assert updated == []

    def test_prune_dead_nodes(self, field):
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "alive"}, phase=0.0, node_id="alive")
        field.add_node(emb, {"text": "dead"}, phase=0.0, node_id="dead")
        field.nodes["dead"].amplitude = 0.01
        field.nodes["dead"].salience = 0.01
        field._prune_dead_nodes()
        assert "alive" in field.nodes
        assert "dead" not in field.nodes

    def test_step_decay(self, field):
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0, node_id="n1")
        initial_amp = field.nodes["n1"].amplitude
        initial_sal = field.nodes["n1"].salience
        field.step()
        assert field.nodes["n1"].amplitude < initial_amp
        assert field.nodes["n1"].salience < initial_sal

    def test_step_creates_node_on_no_resonance(self, field):
        emb = np.random.randn(768).astype(np.float32)
        initial_count = len(field.nodes)
        field.step(inputs=[{"embedding": emb, "phase": 0.0, "content": {"text": "new"}}])
        assert len(field.nodes) == initial_count + 1

    def test_step_resonates_existing_node(self, field):
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "existing"}, phase=0.0, node_id="n1")
        field.nodes["n1"].amplitude = 0.8
        field.nodes["n1"].salience = 0.7
        initial_pos = field.nodes["n1"].latent_pos.copy()
        perturbed = emb + np.random.randn(768).astype(np.float32) * 0.01
        field.step(inputs=[{"embedding": perturbed, "phase": 0.0, "content": {}}])
        assert not np.allclose(field.nodes["n1"].latent_pos, initial_pos)

    def test_max_nodes_limit(self, config):
        config.max_nodes = 5
        field = RTMDKField(config)
        for i in range(10):
            emb = np.random.randn(768).astype(np.float32)
            field.add_node(emb, {"text": f"node_{i}"}, phase=0.0)
        field.step()
        assert len(field.nodes) <= 5

    def test_stats_tracking(self, field):
        emb = np.random.randn(768).astype(np.float32)
        field.add_node(emb, {"text": "test"}, phase=0.0)
        field.query(emb, phase=0.0)
        assert field.stats["total_adds"] == 1
        assert field.stats["total_queries"] == 1


# ============================================================================
# RTMDKMemory TESTS (LangChain Integration)
# ============================================================================

class TestRTMDKMemory:
    def test_memory_variables(self, memory):
        assert memory.memory_variables == ["rtmdk_context"]

    def test_save_context(self, memory):
        inputs = {"input": "Hello world", "session_id": "test"}
        outputs = {"output": "Response to hello"}
        memory.save_context(inputs, outputs)
        assert len(memory.field.nodes) == 1

    def test_save_empty_context(self, memory):
        inputs = {"input": "", "session_id": "test"}
        outputs = {"output": ""}
        memory.save_context(inputs, outputs)
        assert len(memory.field.nodes) == 0

    def test_load_memory_variables(self, memory):
        inputs = {"input": "Test memory", "session_id": "s1"}
        outputs = {"output": "Stored response"}
        memory.save_context(inputs, outputs)
        result = memory.load_memory_variables({"input": "Stored response", "session_id": "s1"})
        assert "rtmdk_context" in result
        assert len(result["rtmdk_context"]) > 0
        assert result["rtmdk_context"] != "No relevant memory."

    def test_load_empty_query(self, memory):
        result = memory.load_memory_variables({"input": ""})
        assert result["rtmdk_context"] == ""

    def test_no_relevant_memory(self, memory):
        inputs = {"input": "Something about cats", "session_id": "s1"}
        outputs = {"output": "Cats are great"}
        memory.save_context(inputs, outputs)
        result = memory.load_memory_variables({"input": "totally unrelated quantum physics"})
        assert "rtmdk_context" in result

    def test_clear(self, memory):
        memory.save_context(
            {"input": "test", "session_id": "s1"},
            {"output": "response"}
        )
        memory.clear()
        assert len(memory.field.nodes) == 0
        assert len(memory.session_phases) == 0

    def test_session_phase_persistence(self, memory):
        phase1 = memory._get_phase("session_A")
        phase2 = memory._get_phase("session_A")
        assert phase1 == phase2

    def test_different_sessions_different_phases(self, memory):
        phase_a = memory._get_phase("session_A")
        time.sleep(0.02)
        phase_b = memory._get_phase("session_B")
        assert phase_a != phase_b

    def test_get_stats(self, memory):
        memory.save_context(
            {"input": "test", "session_id": "s1"},
            {"output": "response"}
        )
        stats = memory.get_stats()
        assert "total_adds" in stats
        assert "total_queries" in stats
        assert "config" in stats

    def test_export_and_import(self, memory, tmp_path, dummy_embedder):
        memory.save_context(
            {"input": "export test", "session_id": "s1"},
            {"output": "exported response"}
        )
        export_path = str(tmp_path / "test_export.json")
        memory.export_field(export_path)
        assert os.path.exists(export_path)
        imported = RTMDKMemory.import_field(export_path, dummy_embedder)
        assert len(imported.field.nodes) == len(memory.field.nodes)
        assert imported.field.stats["total_adds"] == memory.field.stats["total_adds"]

    def test_multiple_saves(self, memory):
        for i in range(5):
            memory.save_context(
                {"input": f"Message {i}", "session_id": "s1"},
                {"output": f"Response {i}"}
            )
        assert len(memory.field.nodes) == 5

    def test_conversation_flow(self, memory):
        interactions = [
            {"input": "Я люблю кофе", "output": "Кофе помогает проснуться.", "session_id": "u1"},
            {"input": "Кофе вреден", "output": "В умеренных количествах безопасен.", "session_id": "u1"},
            {"input": "Я перешёл на чай", "output": "Чай — отличная альтернатива.", "session_id": "u1"},
        ]
        for turn in interactions:
            memory.save_context(turn, turn)
        ctx = memory.load_memory_variables({"input": "Кофе помогает проснуться.", "session_id": "u1"})
        assert "rtmdk_context" in ctx
        assert ctx["rtmdk_context"] != "No relevant memory."


# ============================================================================
# CONSOLIDATION MODE TESTS
# ============================================================================

class TestConsolidationModes:
    def test_dialectical_mode(self, config):
        config.consolidation_mode = ConsolidationMode.DIALECTICAL
        assert config.consolidation_mode == ConsolidationMode.DIALECTICAL

    def test_merge_mode(self, config):
        config.consolidation_mode = ConsolidationMode.MERGE
        assert config.consolidation_mode == ConsolidationMode.MERGE

    def test_prune_mode(self, config):
        config.consolidation_mode = ConsolidationMode.PRUNE
        assert config.consolidation_mode == ConsolidationMode.PRUNE

    def test_consolidate_with_high_tension(self, config):
        """Создаём узлы с высоким напряжением и проверяем консолидацию."""
        config.tension_threshold = 0.05
        config.consolidation_mode = ConsolidationMode.DIALECTICAL
        field = RTMDKField(config)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "a"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "b"}, phase=np.pi, node_id="b")
        updated = field.consolidate()
        assert len(field.nodes) <= 2

    def test_consolidate_merge_mode(self, config):
        config.tension_threshold = 0.05
        config.consolidation_mode = ConsolidationMode.MERGE
        field = RTMDKField(config)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "a"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "b"}, phase=np.pi, node_id="b")
        updated = field.consolidate()
        assert len(field.nodes) <= 2

    def test_consolidate_prune_mode(self, config):
        config.tension_threshold = 0.05
        config.consolidation_mode = ConsolidationMode.PRUNE
        field = RTMDKField(config)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "a"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "b"}, phase=np.pi, node_id="b")
        updated = field.consolidate()
        assert len(field.nodes) <= 2


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    def test_query_top_k_limit(self, field):
        for i in range(10):
            emb = np.random.randn(768).astype(np.float32)
            field.add_node(emb, {"text": f"n{i}"}, phase=0.0)
        emb = np.zeros(768, dtype=np.float32)
        results = field.query(emb, phase=0.0, top_k=3)
        assert len(results) <= 3

    def test_node_lineage_after_consolidation(self, config):
        config.tension_threshold = 0.05
        field = RTMDKField(config)
        emb = np.zeros(768, dtype=np.float32)
        field.add_node(emb, {"text": "a"}, phase=0.0, node_id="a")
        field.add_node(emb, {"text": "b"}, phase=np.pi, node_id="b")
        field.consolidate()
        for node in field.nodes.values():
            if node.lineage:
                assert "+" in node.lineage[0]

    def test_import_nonexistent_file(self, dummy_embedder):
        with pytest.raises(FileNotFoundError):
            RTMDKMemory.import_field("nonexistent_file.json", dummy_embedder)

    def test_projection_2d_input(self, field):
        emb = np.random.randn(1, 768).astype(np.float32)
        latent = field._project(emb)
        assert latent.shape == (1, 64)
