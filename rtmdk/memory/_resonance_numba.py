"""Numba-accelerated resonance kernels (optional — falls back to numpy if numba unavailable)."""

from __future__ import annotations

import math

try:
    import numba  # noqa: F401  (availability probe; njit/prange imported below)
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

    # Dummy decorators for type-checking compatibility
    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn

        return wrapper

    def prange(*args):
        return range(*args)


import numpy as np
from numpy.typing import NDArray


@njit(fastmath=True, cache=True, parallel=True)
def _chunk_resonance_scalar_bw(
    positions: NDArray,
    phases: NDArray,
    amplitudes: NDArray,
    saliences: NDArray,
    modal_weights: NDArray,
    gates: NDArray,
    causal_boost: NDArray,
    query_latent: NDArray,
    query_phase: float,
    bw: float,
    pc: float,
    use_gates: bool,
    use_causal: bool,
) -> NDArray:
    """Numba kernel for chunk resonance with scalar bandwidth (parallel)."""
    n = positions.shape[0]
    dim = positions.shape[1]
    out = np.empty(n, dtype=np.float32)
    bw_sq = bw * bw
    for i in prange(n):
        dist_sq = 0.0
        for d in range(dim):
            diff = positions[i, d] - query_latent[d]
            dist_sq += diff * diff
        spatial = math.exp(-dist_sq / (2.0 * bw_sq))
        phase_align = 0.5 + 0.5 * math.cos(phases[i] - query_phase)
        resp = spatial * ((1.0 - pc) + pc * phase_align)
        resp *= amplitudes[i] * saliences[i] * modal_weights[i]
        if use_gates:
            resp *= gates[i]
        if use_causal:
            resp *= causal_boost[i]
        out[i] = resp
    return out


def chunk_resonance(
    positions: NDArray,
    phases: NDArray,
    amplitudes: NDArray,
    saliences: NDArray,
    modal_weights: NDArray,
    gates: NDArray,
    causal_boost: NDArray,
    query_latent: NDArray,
    query_phase: float,
    bw: float,
    pc: float,
    use_gates: bool,
    use_causal: bool,
) -> NDArray:
    """Dispatch to numba kernel if available, else numpy fallback."""
    if _NUMBA_AVAILABLE:
        return _chunk_resonance_scalar_bw(
            positions,
            phases,
            amplitudes,
            saliences,
            modal_weights,
            gates,
            causal_boost,
            query_latent,
            query_phase,
            bw,
            pc,
            use_gates,
            use_causal,
        )
    # Pure-numpy fallback
    dists = np.linalg.norm(positions - query_latent, axis=1)
    spatial = np.exp(-(dists**2) / (2.0 * bw * bw))
    phase_align = 0.5 + 0.5 * np.cos(phases - query_phase)
    resp = spatial * ((1.0 - pc) + pc * phase_align)
    resp *= amplitudes * saliences * modal_weights
    if use_gates:
        resp *= gates
    if use_causal:
        resp *= causal_boost
    return resp.astype(np.float32)


def is_numba_available() -> bool:
    return _NUMBA_AVAILABLE
