"""Unit tests for ResonanceEngine."""
import numpy as np
import pytest

from rtmdk.memory.resonance import ResonanceEngine
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.quantization import QuantizationHelper


def _engine(bandwidth=1.0, phase_coupling=0.0, hyperbolic=False):
    cfg = RTMDKConfig(
        latent_dim=8,
        bandwidth=bandwidth,
        phase_coupling=phase_coupling,
        hyperbolic=hyperbolic,
    )
    quant = QuantizationHelper("none")
    return ResonanceEngine(cfg, quant=quant)


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
        self.modality = "text"


def test_single_response_identity():
    engine = _engine(bandwidth=1.0, phase_coupling=0.0)
    pos = np.ones(8, dtype=np.float32)
    query = np.ones(8, dtype=np.float32)
    node = FakeNode(pos)
    score = engine.single_response(query, 0.0, node)
    # Same vector -> dist 0 -> spatial=1.0, phase_align=1.0 -> resp=1.0
    assert pytest.approx(score, 0.01) == 1.0


def test_single_response_phase_coupling():
    engine = _engine(bandwidth=1.0, phase_coupling=1.0)
    pos = np.ones(8, dtype=np.float32)
    query = np.ones(8, dtype=np.float32)
    node = FakeNode(pos, phase=np.pi)
    score = engine.single_response(query, 0.0, node)
    # phase diff = pi -> phase_align = 0.0 -> resp = 0.0
    assert pytest.approx(score, 0.01) == 0.0


def test_batch_response_numpy():
    engine = _engine(bandwidth=1.0, phase_coupling=0.0)
    positions = np.ones((3, 8), dtype=np.float32)
    phases = np.zeros(3, dtype=np.float32)
    query_latents = np.ones((1, 8), dtype=np.float32)
    query_phases = np.zeros(1, dtype=np.float32)
    amps = np.ones(3, dtype=np.float32)
    sals = np.ones(3, dtype=np.float32)

    out = engine.batch_response_numpy(
        query_latents, query_phases, positions, phases, amps, sals)
    assert out.shape == (1, 3)
    assert np.allclose(out, 1.0)


def test_chunk_response():
    engine = _engine(bandwidth=1.0, phase_coupling=0.3)
    positions = np.random.randn(100, 8).astype(np.float32)
    phases = np.random.randn(100).astype(np.float32)
    amps = np.ones(100, dtype=np.float32)
    sals = np.ones(100, dtype=np.float32)
    mw = np.ones(100, dtype=np.float32)
    gates = np.ones(100, dtype=np.float32)
    cboost = np.ones(100, dtype=np.float32)
    query = np.random.randn(8).astype(np.float32)

    out = engine.chunk_response(
        positions, phases, amps, sals, mw, gates, cboost,
        query, 0.0, bw=1.0, use_gates=False, use_causal=False)
    assert out.shape == (100,)
    assert out.dtype == np.float32
