"""rtmdk/memory/quantization.py — Embedding quantization helpers.

Supported modes:
  - none   : float32 (no quantization)
  - fp16   : 16-bit float  → ~2× RAM reduction, ~100% recall
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class QuantizationHelper:
    """Quantize / dequantize latent position vectors."""

    def __init__(self, mode: str):
        self.mode = mode

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dtype(self):
        if self.mode == "fp16":
            return np.float16
        return np.float32

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize  # 2 for fp16, 4 for fp32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def quantize(self, vec: NDArray) -> NDArray:
        """Return quantized copy (or original if mode == 'none')."""
        if self.mode == "fp16":
            return vec.astype(np.float16)
        return vec.astype(np.float32)

    def dequantize(self, qvec: NDArray) -> NDArray:
        """Return float32 copy (or original if mode == 'none')."""
        if self.mode == "fp16":
            return qvec.astype(np.float32)
        return qvec.astype(np.float32)

    def maybe_dequantize(self, qvec: NDArray) -> NDArray:
        """Dequantize only if currently quantized."""
        if self.mode == "none":
            return qvec
        return self.dequantize(qvec)
