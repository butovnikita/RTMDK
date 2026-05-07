"""rtmdk/production/gpu_backend.py — GPU offload stub for RTMDK.

Future: CUDA-accelerated resonance computation and batch embeddings.
"""

import numpy as np


class GPUBackend:
    """Stub for GPU-accelerated operations.

    When torch+cuda is available, this class can offload:
    - Batch resonance computation
    - Embedding projections
    - HNSW index updates
    """

    def __init__(self):
        self._cuda_available = False
        self._torch = None
        try:
            import torch
            self._cuda_available = torch.cuda.is_available()
            self._torch = torch
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._cuda_available

    def to_device(self, arr: np.ndarray):
        """Move numpy array to GPU if available."""
        if not self.available or self._torch is None:
            return arr
        return self._torch.from_numpy(arr).cuda()

    def batch_resonance(self, query: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Compute distances on GPU (stub — falls back to CPU)."""
        if not self.available:
            import numpy as np
            return np.linalg.norm(positions - query, axis=1)
        t_query = self.to_device(query)
        t_positions = self.to_device(positions)
        dists = self._torch.norm(t_positions - t_query, dim=1)
        return dists.cpu().numpy()
