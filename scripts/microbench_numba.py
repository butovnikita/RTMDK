"""Microbenchmark Numba vs Numpy resonance chunk."""

import time
import numpy as np
from rtmdk.memory._resonance_numba import chunk_resonance as chunk_numba


def chunk_numpy(
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
):
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


def bench(n, dim=384, repeats=20, warmup=5):
    rng = np.random.default_rng(42)
    positions = rng.standard_normal((n, dim), dtype=np.float32)
    # normalize for realistic distances
    positions /= np.linalg.norm(positions, axis=1, keepdims=True) + 1e-8
    phases = rng.random(n, dtype=np.float32)
    amps = rng.random(n, dtype=np.float32)
    sal = rng.random(n, dtype=np.float32)
    mw = rng.random(n, dtype=np.float32)
    gates = rng.random(n, dtype=np.float32)
    cboost = rng.random(n, dtype=np.float32)
    q = rng.standard_normal(dim, dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-8
    qp = 0.5
    bw = 1.0
    pc = 0.3

    # Warmup numba (compile)
    for _ in range(warmup):
        _ = chunk_numba(positions, phases, amps, sal, mw, gates, cboost, q, qp, bw, pc, False, False)

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = chunk_numpy(positions, phases, amps, sal, mw, gates, cboost, q, qp, bw, pc, False, False)
    t_numpy = (time.perf_counter() - t0) / repeats * 1000

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = chunk_numba(positions, phases, amps, sal, mw, gates, cboost, q, qp, bw, pc, False, False)
    t_numba = (time.perf_counter() - t0) / repeats * 1000

    print(f"n={n:>7}  numpy={t_numpy:>7.3f}ms  numba={t_numba:>7.3f}ms  ratio={t_numpy/t_numba:>5.2f}x")


if __name__ == "__main__":
    for n in [5000, 20000, 50000, 100000, 200000]:
        bench(n)
