"""ResonanceEngine — pure resonance computation extracted from RTMDKField.

Provides single-node and batch resonance scoring without cache or index
management.  All numpy/torch math lives here; RTMDKField delegates resonance
calls to this engine.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

from rtmdk.memory.geometry import poincare_dist
from rtmdk.memory.utils import cross_modal_resonance
from rtmdk.memory._resonance_numba import chunk_resonance as _chunk_resonance_numba, is_numba_available


class ResonanceEngine:
    """Encapsulates resonance math: spatial × phase × amplitude × salience."""

    def __init__(
        self,
        cfg,
        meta_kernel: Optional[Any] = None,
        learnable_kernel: Optional[Any] = None,
        causal_engine: Optional[Any] = None,
        gpu_backend: Optional[Any] = None,
        quant: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.meta_kernel = meta_kernel
        self.learnable_kernel = learnable_kernel
        self.causal_engine = causal_engine
        self.gpu_backend = gpu_backend
        self._quant = quant

        # Backward-compat: stats accumulator for hyperbolic distance
        self.stats: dict = {"avg_hyperbolic_dist": 0.0}

    # ------------------------------------------------------------------
    # Helpers that mirror RTMDKField properties
    # ------------------------------------------------------------------
    @property
    def _effective_bandwidth(self) -> float:
        if self.meta_kernel is not None and getattr(
                self.meta_kernel, "_best_bw", None) is not None:
            return self.meta_kernel._best_bw
        return self.cfg.bandwidth

    @property
    def _effective_pc(self) -> float:
        if self.cfg.learnable_phase_coupling and self.learnable_kernel is not None:
            return getattr(self.learnable_kernel, "phase_coupling", self.cfg.phase_coupling)
        return self.cfg.phase_coupling

    # ------------------------------------------------------------------
    # Single-node response
    # ------------------------------------------------------------------
    def single_response(
        self,
        query_latent: NDArray,
        query_phase: float,
        node: Any,
        query_modality: str = "text",
    ) -> float:
        """Compute resonance for one query vs one node."""
        if self.cfg.hyperbolic:
            dist = poincare_dist(
                query_latent, node.latent_pos, self.cfg.ball_radius)
        else:
            dist = np.linalg.norm(query_latent - node.latent_pos)

        phase_diff = node.phase - query_phase
        bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self._effective_bandwidth
        bw = max(bw, 1e-8)
        pc = self._effective_pc

        if self.learnable_kernel is not None:
            resp = self.learnable_kernel.resonance_response(
                dist, phase_diff, node.amplitude, node.salience)
        else:
            if self.cfg.resonance_kernel in ("gaussian", "gaussian_phase"):
                spatial = math.exp(-dist ** 2 / (2 * bw ** 2))
            elif self.cfg.resonance_kernel == "cosine":
                nq = np.linalg.norm(query_latent)
                nn = np.linalg.norm(node.latent_pos)
                if nq > 1e-8 and nn > 1e-8:
                    spatial = 0.5 + 0.5 * np.dot(query_latent, node.latent_pos) / (nq * nn)
                else:
                    spatial = 0.5
            else:
                spatial = math.exp(-dist / bw)
            phase_align = 0.5 + 0.5 * math.cos(phase_diff)
            resp = spatial * ((1 - pc) + pc * phase_align) * \
                node.amplitude * node.salience

        gate = node.soft_gate if self.cfg.soft_gates else 1.0
        if self.causal_engine and getattr(node, "causal_parents", None):
            causal_boost = sum(node.causal_strength.get(p, 0)
                               for p in node.causal_parents)
            resp *= (1.0 + 0.1 * causal_boost)

        if self.cfg.cross_modal:
            resp = cross_modal_resonance(
                query_modality,
                node.modality,
                resp,
                self.cfg.modal_phase_offsets,
                self.cfg.cross_modal_kernel_weight)
            base_val = spatial * node.amplitude * node.salience
            node.cross_modal_score = resp / base_val if base_val > 1e-8 else 0.0

        return resp * gate * node.modal_weight

    # ------------------------------------------------------------------
    # Batch response (numpy)
    # ------------------------------------------------------------------
    def batch_response_numpy(
        self,
        query_latents: NDArray,
        query_phases: NDArray,
        node_positions: NDArray,
        node_phases: NDArray,
        node_amplitudes: NDArray,
        node_saliences: NDArray,
        bw: Optional[float] = None,
        pc: Optional[float] = None,
    ) -> NDArray:
        """Vectorized resonance over pre-materialized node arrays.

        All inputs must be numpy arrays of matching first dimension.
        """
        if node_positions.shape[0] == 0:
            return np.empty((len(query_latents), 0), dtype=np.float32)

        dists = cdist(query_latents, node_positions)
        if bw is None:
            bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self._effective_bandwidth
        bw = np.maximum(bw, 1e-8)
        if pc is None:
            pc = self._effective_pc

        if np.ndim(bw) == 0:
            spatial = np.exp(-dists ** 2 / (2 * bw ** 2))
        else:
            spatial = np.exp(-dists ** 2 / (2 * bw[np.newaxis, :] ** 2))

        phase_diff = query_phases[:, np.newaxis] - node_phases[np.newaxis, :]
        phase_align = 0.5 + 0.5 * np.cos(phase_diff)
        response = spatial * ((1 - pc) + pc * phase_align)
        return response * node_amplitudes[np.newaxis, :] * node_saliences[np.newaxis, :]

    # ------------------------------------------------------------------
    # Batch response (torch / GPU)
    # ------------------------------------------------------------------
    def batch_response_torch(
        self,
        query_latents: NDArray,
        query_phases: NDArray,
        node_positions: NDArray,
        node_phases: NDArray,
        node_amplitudes: NDArray,
        node_saliences: NDArray,
    ) -> NDArray:
        """GPU-accelerated batch resonance via TorchBackend."""
        if node_positions.shape[0] == 0:
            return np.empty((len(query_latents), 0), dtype=np.float32)
        if self.gpu_backend is None:
            raise RuntimeError("GPU backend not available")
        bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self.cfg.bandwidth
        pc = self._effective_pc
        return self.gpu_backend.batch_resonance(
            query_latents, query_phases,
            node_positions, node_phases,
            node_amplitudes, node_saliences,
            bw, pc,
        )

    # ------------------------------------------------------------------
    # Chunk resonance (used by _query_vectorized)
    # ------------------------------------------------------------------
    def chunk_response(
        self,
        positions: NDArray,
        phases: NDArray,
        amplitudes: NDArray,
        saliences: NDArray,
        modal_weights: NDArray,
        gates: NDArray,
        causal_boost: NDArray,
        query_latent: NDArray,
        query_phase: float,
        bw: Optional[Any] = None,
        use_gates: bool = False,
        use_causal: bool = False,
    ) -> NDArray:
        """Compute resonance for a chunk of nodes (used by _query_vectorized)."""
        if bw is not None:
            local_bw = float(np.asarray(bw))
        else:
            local_bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self.cfg.bandwidth
        local_bw = max(local_bw, 1e-8)
        pc = float(self._effective_pc)

        # Numba fast-path for scalar bandwidth (the common case)
        if is_numba_available() and np.ndim(bw) == 0:
            # zero-copy only when dtype already matches; otherwise cast
            def _asf32(a):
                a = np.asarray(a)
                return a if a.dtype == np.float32 else a.astype(np.float32)
            return _chunk_resonance_numba(
                _asf32(positions),
                _asf32(phases),
                _asf32(amplitudes),
                _asf32(saliences),
                _asf32(modal_weights),
                _asf32(gates),
                _asf32(causal_boost),
                _asf32(query_latent),
                float(query_phase),
                local_bw,
                pc,
                use_gates,
                use_causal,
            )

        # Pure-numpy fallback (supports per-node bw vectors)
        dists = np.linalg.norm(positions - query_latent, axis=1)
        if np.ndim(local_bw) == 0:
            spatial = np.exp(-dists ** 2 / (2 * local_bw ** 2))
        else:
            spatial = np.exp(-dists ** 2 / (2 * local_bw ** 2))
        phase_align = 0.5 + 0.5 * np.cos(phases - query_phase)
        resp = spatial * ((1 - pc) + pc * phase_align)
        resp *= amplitudes * saliences * modal_weights
        if use_gates:
            resp *= gates
        if use_causal:
            resp *= causal_boost
        return resp.astype(np.float32)
