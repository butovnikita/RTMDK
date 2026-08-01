"""rtmdk/production/gpu_backend.py — GPU offload façade.

Delegates to the full-featured ``GPUBackend`` in ``rtmdk/support/triton_backend.py``
when available, and provides additional convenience helpers for batch distance
computation and embedding projection.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

# Re-export the canonical backend from support (already production-ready)
try:
    from rtmdk.support.triton_backend import (
        GPUBackend as _GPUBackend,
        TRITON_AVAILABLE,
        _numpy_resonance,
    )
except Exception:  # pragma: no cover
    _GPUBackend = None  # type: ignore[assignment,misc]
    TRITON_AVAILABLE = False


class GPUBackend:
    """Production GPU backend façade.

    Wraps ``rtmdk.support.triton_backend.GPUBackend`` and adds:
    - ``batch_resonance`` (dense/sparse) — full resonance kernel
    - ``batch_distance`` — fast L2 distance for candidate pruning
    - ``project`` — linear projection on GPU
    """

    def __init__(self, min_nodes_for_gpu: int = 2000):
        self.min_nodes_for_gpu = min_nodes_for_gpu
        self._inner: Optional[Any] = None
        if _GPUBackend is not None:
            try:
                self._inner = _GPUBackend(min_nodes_for_gpu=min_nodes_for_gpu)
            except Exception:
                pass

    @property
    def available(self) -> bool:
        if self._inner is not None:
            return bool(getattr(self._inner, "available", False))
        return False

    def should_use_gpu(self, n_nodes: int, n_queries: int = 1) -> bool:
        if self._inner is not None:
            return self._inner.should_use_gpu(n_nodes, n_queries)
        return False

    def batch_resonance(
        self,
        query_latents: NDArray,
        query_phases: NDArray,
        node_positions: NDArray,
        node_phases: NDArray,
        node_amplitudes: NDArray,
        node_saliences: NDArray,
        bandwidth: float,
        phase_coupling: float,
        candidate_mask: Optional[NDArray] = None,
    ) -> NDArray:
        """Compute batch resonance with auto-backend selection."""
        if self._inner is not None:
            return self._inner.batch_resonance(
                query_latents,
                query_phases,
                node_positions,
                node_phases,
                node_amplitudes,
                node_saliences,
                bandwidth,
                phase_coupling,
                candidate_mask,
            )
        return _numpy_resonance(
            query_latents,
            query_phases,
            node_positions,
            node_phases,
            node_amplitudes,
            node_saliences,
            bandwidth,
            phase_coupling,
            candidate_mask,
        )

    def batch_distance(self, query: NDArray, positions: NDArray) -> NDArray:
        """Fast L2 distances: ``||positions - query||_2``."""
        if not self.available:
            return np.linalg.norm(positions - query, axis=1)
        import torch

        t_query = torch.from_numpy(query).cuda()
        t_positions = torch.from_numpy(positions).cuda()
        return torch.norm(t_positions - t_query, dim=1).cpu().numpy()

    def project(self, vectors: NDArray, matrix: NDArray) -> NDArray:
        """Linear projection ``vectors @ matrix.T`` on GPU if beneficial."""
        n = vectors.shape[0]
        if self.should_use_gpu(n):
            import torch

            t_v = torch.from_numpy(vectors).cuda()
            t_m = torch.from_numpy(matrix).cuda()
            return (t_v @ t_m.T).cpu().numpy()
        return vectors @ matrix.T


# Backward-compatible alias
TritonBackend = GPUBackend
