"""Tests for rtmdk.support.tda."""

import numpy as np
from rtmdk.support.tda import TDAMonitor
from rtmdk.nodes import MemoryNode


def _make_node(pos):
    node = MemoryNode.__new__(MemoryNode)
    node.latent_pos = np.array(pos, dtype=np.float32)
    return node


class TestTDAMonitor:
    def test_persistence_too_few_nodes(self):
        monitor = TDAMonitor()
        result = monitor.compute_persistence({})
        assert result == {"H0": 0, "H1": 0, "avg_persistence": 0.0}

    def test_persistence_basic(self):
        monitor = TDAMonitor()
        nodes = {
            "a": _make_node([0.0, 0.0]),
            "b": _make_node([1.0, 0.0]),
            "c": _make_node([0.5, 1.0]),
        }
        result = monitor.compute_persistence(nodes)
        assert "H0" in result
        assert "H1" in result
        assert len(monitor.history) == 1

    def test_get_trend_stable(self):
        monitor = TDAMonitor()
        assert monitor.get_trend() == "stable"

    def test_get_trend_growing(self):
        monitor = TDAMonitor()
        for i in range(5):
            monitor.history.append({"H0": 1, "H1": i + 1})
        assert monitor.get_trend() == "growing_contradictions"
