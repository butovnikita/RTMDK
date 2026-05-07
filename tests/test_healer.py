"""Tests for rtmdk.support.healer."""

import numpy as np
import pytest
from rtmdk.support.healer import TopologyHealer
from rtmdk.nodes import MemoryNode


def _make_node(pos, salience=0.5, amplitude=0.5):
    node = MemoryNode.__new__(MemoryNode)
    node.latent_pos = np.array(pos, dtype=np.float32)
    node.salience = salience
    node.amplitude = amplitude
    node.is_healing = False
    node.healing_origin = None
    return node


class TestTopologyHealer:
    def test_detect_dead_zones_too_few(self):
        healer = TopologyHealer()
        assert healer.detect_dead_zones({}) == []

    def test_detect_dead_zones(self):
        healer = TopologyHealer()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([0.1, 0.0]),
            "c": _make_node([10.0, 10.0]),
        }
        dead = healer.detect_dead_zones(nodes)
        assert "c" in dead

    def test_detect_hyperconvergence_false(self):
        healer = TopologyHealer()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([1.0, 0.0]),
            "c": _make_node([0.0, 1.0]),
        }
        assert healer.detect_hyperconvergence(nodes) is False

    def test_detect_fragmentation(self):
        healer = TopologyHealer()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([100.0, 100.0]),
        }
        frag = healer.detect_fragmentation(nodes, radius=2.0)
        assert frag == pytest.approx(1.0)

    def test_compute_field_health(self):
        healer = TopologyHealer()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([1.0, 0.0]),
            "c": _make_node([0.0, 1.0]),
        }
        health, diag = healer.compute_field_health(nodes)
        assert health.value == "stable"
        assert "dead_zones" in diag

    def test_heal_dead_zones(self):
        healer = TopologyHealer()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([10.0, 10.0]),
        }
        healed = healer.heal_dead_zones(nodes, ["b"])
        assert len(healed) == 1
        assert healed[0]["node_id"] == "b"

    def test_heal_hyperconvergence(self):
        healer = TopologyHealer()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([0.01, 0.0]),
            "c": _make_node([0.0, 0.01]),
        }
        healed = healer.heal_hyperconvergence(nodes)
        assert len(healed) > 0
        assert healed[0]["type"] == "hyperconvergence"

    def test_heal_fragmentation(self):
        healer = TopologyHealer()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([100.0, 100.0]),
        }
        healed = healer.heal_fragmentation(nodes, ["b"])
        assert len(healed) == 1
        assert healed[0]["type"] == "fragmentation"

    def test_state_roundtrip(self):
        healer = TopologyHealer()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([1.0, 0.0]),
            "c": _make_node([0.0, 1.0]),
        }
        healer.compute_field_health(nodes)
        state = healer.get_state()
        healer2 = TopologyHealer()
        healer2.load_state(state)
        assert healer2.get_state()["health_history"] == state["health_history"]
