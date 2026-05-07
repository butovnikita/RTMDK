"""
tests/test_nodes.py — Unit tests for RTMDK node data classes.

Covers:
1. MemoryNode serialization round-trip
2. CausalEdge serialization
3. ContradictionRecord serialization
"""

import numpy as np

from rtmdk.nodes import MemoryNode, CausalEdge, ContradictionRecord


class TestMemoryNode:
    def test_to_dict_from_dict_roundtrip(self):
        node = MemoryNode(
            id="test_1",
            latent_pos=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            phase=0.5,
            amplitude=0.8,
            salience=0.9,
            content={"text": "hello world"},
        )
        d = node.to_dict()
        assert d["id"] == "test_1"
        assert np.allclose(d["latent_pos"], [0.1, 0.2, 0.3])
        assert d["phase"] == 0.5
        assert d["content"]["text"] == "hello world"

        restored = MemoryNode.from_dict(d)
        assert restored.id == node.id
        assert np.allclose(restored.latent_pos, node.latent_pos)
        assert restored.phase == node.phase
        assert restored.amplitude == node.amplitude
        assert restored.salience == node.salience

    def test_optional_fields_roundtrip(self):
        node = MemoryNode(
            id="test_2",
            latent_pos=np.array([1.0, 2.0], dtype=np.float32),
            phase=1.0,
            amplitude=1.0,
            salience=1.0,
            pre_consolidation_pos=np.array([0.5, 1.5], dtype=np.float32),
            gradient_cache=np.array([0.1, 0.1], dtype=np.float32),
            velocity=np.array([0.01, 0.02], dtype=np.float32),
            acceleration=np.array([0.001, 0.002], dtype=np.float32),
            modal_embedding=np.array([0.5, 0.5], dtype=np.float32),
            do_interventions={"test": np.array([1.0, 2.0], dtype=np.float32)},
        )
        d = node.to_dict()
        restored = MemoryNode.from_dict(d)
        assert np.allclose(
            restored.pre_consolidation_pos,
            node.pre_consolidation_pos)
        assert np.allclose(restored.gradient_cache, node.gradient_cache)
        assert np.allclose(restored.velocity, node.velocity)
        assert np.allclose(restored.acceleration, node.acceleration)
        assert np.allclose(restored.modal_embedding, node.modal_embedding)
        assert np.allclose(
            restored.do_interventions["test"],
            node.do_interventions["test"])

    def test_default_values(self):
        node = MemoryNode(
            id="test_3",
            latent_pos=np.zeros(64, dtype=np.float32),
            phase=0.0,
            amplitude=0.5,
            salience=0.5,
        )
        assert node.tension == 0.0
        assert node.soft_gate == 1.0
        assert node.modality == "text"
        assert node.tier == "semantic"
        assert node.role == "default"


class TestCausalEdge:
    def test_to_dict_from_dict(self):
        edge = CausalEdge(
            source="n1",
            target="n2",
            strength=0.8,
            confidence=0.9,
        )
        d = edge.to_dict()
        assert d["source"] == "n1"
        assert d["target"] == "n2"
        restored = CausalEdge.from_dict(d)
        assert restored.source == edge.source
        assert restored.strength == edge.strength


class TestContradictionRecord:
    def test_to_dict_from_dict(self):
        rec = ContradictionRecord(
            id="c1",
            effect_node="n3",
            causes=[("n1", 0.7), ("n2", 0.3)],
        )
        d = rec.to_dict()
        assert d["effect_node"] == "n3"
        assert d["causes"] == [("n1", 0.7), ("n2", 0.3)]
        assert d["resolved"] is False
