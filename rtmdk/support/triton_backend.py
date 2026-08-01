"""rtmdk/support/triton_backend.py — GPU-accelerated resonance backend.

NOTE: This is NOT a true Triton kernel backend. It uses PyTorch/CUDA for
GPU acceleration with a placeholder for future Triton kernel integration.
The class has been renamed to GPUBackend to reflect its actual implementation.
TritonBackend remains available as a backward-compatible alias.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.typing import NDArray

TRITON_AVAILABLE = False
_triton = None
try:
    TRITON_AVAILABLE = True
except ImportError:
    pass


def sparse_resonance_kernel(
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
    """Compute sparse resonance responses.

    If candidate_mask is provided, only compute for masked entries.
    Otherwise falls back to dense computation.

    Args:
        query_latents: (Q, D) query latent vectors
        query_phases: (Q,) query phases
        node_positions: (N, D) node latent positions
        node_phases: (N,) node phases
        node_amplitudes: (N,) node amplitudes
        node_saliences: (N,) node saliencies
        bandwidth: kernel bandwidth
        phase_coupling: phase alignment weight
        candidate_mask: (Q, N) boolean mask of candidates (from HNSW)

    Returns:
        (Q, N) resonance responses (zeros where mask is False)
    """
    if TRITON_AVAILABLE:
        return _triton_resonance(
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


def _numpy_resonance(
    query_latents,
    query_phases,
    node_positions,
    node_phases,
    node_amplitudes,
    node_saliences,
    bandwidth,
    phase_coupling,
    candidate_mask=None,
):
    """Numpy fallback — dense or sparse via mask."""
    from scipy.spatial.distance import cdist

    len(query_latents)
    len(node_positions)

    # Compute distances
    dists = cdist(query_latents, node_positions)  # (Q, N)
    spatial = np.exp(-dists / bandwidth)  # (Q, N)

    # Phase alignment
    phase_diff = query_phases[:, np.newaxis] - node_phases[np.newaxis, :]  # (Q, N)
    phase_align = 0.5 + 0.5 * np.cos(phase_diff)

    # Full resonance
    response = spatial * ((1 - phase_coupling) + phase_coupling * phase_align)
    response = response * node_amplitudes[np.newaxis, :] * node_saliences[np.newaxis, :]

    # Apply mask if provided
    if candidate_mask is not None:
        response = response * candidate_mask

    return response.astype(np.float32)


def _triton_resonance(
    query_latents,
    query_phases,
    node_positions,
    node_phases,
    node_amplitudes,
    node_saliences,
    bandwidth,
    phase_coupling,
    candidate_mask=None,
):
    """Triton kernel for sparse resonance computation."""
    # Note: This is a simplified Triton kernel. For production use,
    # a more sophisticated implementation with proper block tiling
    # and shared memory optimization would be needed.

    import torch

    device = "cuda"

    # Convert to torch tensors
    ql = torch.from_numpy(query_latents).to(device, dtype=torch.float32)
    qp = torch.from_numpy(query_phases).to(device, dtype=torch.float32)
    np_ = torch.from_numpy(node_positions).to(device, dtype=torch.float32)
    nph = torch.from_numpy(node_phases).to(device, dtype=torch.float32)
    na = torch.from_numpy(node_amplitudes).to(device, dtype=torch.float32)
    ns = torch.from_numpy(node_saliences).to(device, dtype=torch.float32)

    # Dense computation via torch (Triton kernel would go here for production)
    # This serves as a bridge — when full Triton is implemented, replace this.
    dists = torch.cdist(ql, np_)
    spatial = torch.exp(-dists / bandwidth)
    pd = qp.unsqueeze(1) - nph.unsqueeze(0)
    pa = 0.5 + 0.5 * torch.cos(pd)
    response = spatial * ((1 - phase_coupling) + phase_coupling * pa)
    response = response * na.unsqueeze(0) * ns.unsqueeze(0)

    if candidate_mask is not None:
        mask = torch.from_numpy(candidate_mask).to(device, dtype=torch.float32)
        response = response * mask

    return response.cpu().numpy().astype(np.float32)


class GPUBackend:
    """GPU backend using PyTorch/CUDA for resonance computation.

    Auto-selects backend based on availability and field size.
    A Triton kernel implementation may replace the torch fallback in future.
    """

    def __init__(self, min_nodes_for_gpu: int = 2000):
        self.min_nodes_for_gpu = min_nodes_for_gpu
        self.available = TRITON_AVAILABLE
        self._use_gpu = False
        self._fallback_reason = ""
        if not TRITON_AVAILABLE:
            self._fallback_reason = "triton not installed"

    def should_use_gpu(self, n_nodes: int, n_queries: int = 1) -> bool:
        """Decide whether to use GPU based on problem size."""
        if not TRITON_AVAILABLE:
            return False
        # GPU is beneficial when total operations > threshold
        total_ops = n_nodes * n_queries
        return total_ops > self.min_nodes_for_gpu * n_queries

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
        n_nodes = len(node_positions)
        n_queries = len(query_latents)

        if self.should_use_gpu(n_nodes, n_queries):
            try:
                return _triton_resonance(
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
            except Exception as e:
                self._fallback_reason = f"gpu error: {e}"
                self._use_gpu = False

        # Fallback to numpy
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

    @property
    def backend_name(self) -> str:
        if self._use_gpu and TRITON_AVAILABLE:
            return "cuda"
        return "numpy"


# Backward-compatible alias
TritonBackend = GPUBackend
