"""Verify batch_response_numpy_int8 matches batch_response_numpy numerically."""

import numpy as np

from rtmdk.memory.resonance import ResonanceEngine
from rtmdk.memory.config import RTMDKConfig


def _make_engine():
    cfg = RTMDKConfig(latent_dim=64, bandwidth=1.0, phase_coupling=0.3)
    return ResonanceEngine(cfg)


class TestInt8ResonanceFastPath:
    def test_int8_fast_path_matches_float_reference(self):
        """batch_response_numpy_int8 must match batch_response_numpy on
        dequantized data within floating-point tolerance."""
        engine = _make_engine()

        rng = np.random.RandomState(42)
        nq, nn, dim = 5, 100, 64

        query_latents = rng.randn(nq, dim).astype(np.float32)
        query_latents /= np.linalg.norm(query_latents, axis=1, keepdims=True)
        query_phases = rng.uniform(0, 2 * np.pi, nq).astype(np.float32)

        # Float32 reference positions
        node_positions_f32 = rng.randn(nn, dim).astype(np.float32)
        node_positions_f32 /= np.linalg.norm(node_positions_f32, axis=1, keepdims=True)

        # Quantize to int8 for fast path
        scales = []
        node_positions_int8 = []
        for vec in node_positions_f32:
            max_abs = float(np.abs(vec).max())
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            q = np.round(vec / scale).astype(np.int8)
            node_positions_int8.append(q)
            scales.append(scale)
        node_positions_int8 = np.stack(node_positions_int8)
        scales = np.array(scales, dtype=np.float32)

        # Dequantize back for reference
        node_positions_deq = node_positions_int8.astype(np.float32) * scales[:, np.newaxis]
        node_phases = rng.uniform(0, 2 * np.pi, nn).astype(np.float32)
        node_amplitudes = np.ones(nn, dtype=np.float32)
        node_saliences = np.ones(nn, dtype=np.float32)

        # Reference via float32 cdist
        ref = engine.batch_response_numpy(
            query_latents,
            query_phases,
            node_positions_deq,
            node_phases,
            node_amplitudes,
            node_saliences,
        )

        # Fast path via int8 BLAS matmul
        norms_sq = np.einsum("ij,ij->i", node_positions_deq, node_positions_deq)
        fast = engine.batch_response_numpy_int8(
            query_latents,
            query_phases,
            node_positions_int8,
            norms_sq,
            scales,
            node_phases,
            node_amplitudes,
            node_saliences,
        )

        # Tolerance: int8 roundtrip introduces ~1% relative error
        np.testing.assert_allclose(ref, fast, rtol=0.02, atol=1e-4)

    def test_int8_fast_path_empty_positions(self):
        engine = _make_engine()
        fast = engine.batch_response_numpy_int8(
            np.empty((3, 64), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.empty((0, 64), dtype=np.int8),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
        assert fast.shape == (3, 0)
        assert fast.dtype == np.float32
