"""Unit tests for NodeCacheManager."""

import numpy as np

from rtmdk.memory.cache_manager import NodeCacheManager
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.quantization import QuantizationHelper


class FakeNode:
    def __init__(self, pos, phase=0.0, amp=1.0, sal=1.0, mw=1.0):
        self.latent_pos = np.asarray(pos, dtype=np.float32)
        self.phase = float(phase)
        self.amplitude = float(amp)
        self.salience = float(sal)
        self.modal_weight = float(mw)
        self.soft_gate = 1.0
        self.causal_parents = []
        self.causal_strength = {}
        self.content = {}


class FakeField:
    def __init__(self, nodes, node_index, cfg, quant):
        self.nodes = nodes
        self.node_index = node_index
        self.cfg = cfg
        self._quant = quant
        self._tiered_store = None
        self.causal_engine = None


def test_cache_manager_build_and_lookup():
    cfg = RTMDKConfig(latent_dim=8)
    quant = QuantizationHelper("none")
    nodes = {
        "a": FakeNode(np.ones(8)),
        "b": FakeNode(np.zeros(8)),
    }
    field = FakeField(nodes, ["a", "b"], cfg, quant)
    mgr = NodeCacheManager()
    mgr.build(field)

    assert mgr.positions is not None
    assert mgr.positions.shape == (2, 8)
    assert len(mgr.node_id_to_idx) == 2
    assert mgr.node_id_to_idx["a"] == 0
    assert mgr.node_id_to_idx["b"] == 1
    assert not mgr.dirty


def test_cache_manager_get_indices():
    cfg = RTMDKConfig(latent_dim=8)
    quant = QuantizationHelper("none")
    nodes = {f"n{i}": FakeNode(np.ones(8) * i) for i in range(5)}
    field = FakeField(nodes, [f"n{i}" for i in range(5)], cfg, quant)
    mgr = NodeCacheManager()
    mgr.build(field)

    idx = mgr.get_indices(["n1", "n3"])
    assert idx.tolist() == [1, 3]


def test_cache_manager_invalidate():
    mgr = NodeCacheManager()
    assert not mgr.dirty
    mgr.invalidate()
    assert mgr.dirty
