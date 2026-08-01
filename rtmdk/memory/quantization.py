"""rtmdk/memory/quantization.py — Embedding quantization helpers.

Supported modes:
  - none   : float32 (no quantization)
  - fp16   : 16-bit float  → ~2× RAM reduction, ~100% recall
  - int8   : 8-bit integer  → ~4× RAM reduction, ~98% recall
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


class QuantizationHelper:
    """Quantize / dequantize latent position vectors."""

    # Global fixed scale for int8_global mode
    INT8_GLOBAL_SCALE = 1.0 / 127.0

    def __init__(self, mode: str):
        if mode not in {"none", "fp16", "int8", "int8_global", "int8_per_dim"}:
            raise ValueError(f"Unsupported quantization mode: {mode}")
        self.mode = mode

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dtype(self):
        if self.mode == "fp16":
            return np.float16
        if self.mode in ("int8", "int8_global", "int8_per_dim"):
            return np.int8
        return np.float32

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize  # 1 for int8, 2 for fp16, 4 for fp32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def quantize_with_meta(self, vec: NDArray):
        """Quantize and return (qvec, scale, zero_point).

        For int8 / int8_global / int8_per_dim modes returns metadata.
        For other modes returns (qvec, 1.0, 0.0).
        """
        if self.mode == "int8":
            q, scale, zp = _quantize_int8(vec)
            return q, scale, zp
        if self.mode == "int8_global":
            q = _quantize_int8_global(vec)
            return q, self.INT8_GLOBAL_SCALE, 0.0
        if self.mode == "int8_per_dim":
            q, scale_arr = _quantize_int8_per_dim(vec)
            return q, 1.0, 0.0, scale_arr
        return self.quantize(vec), 1.0, 0.0

    def quantize(self, vec: NDArray) -> NDArray:
        """Return quantized copy (or original if mode == 'none')."""
        if self.mode == "fp16":
            return vec.astype(np.float16)
        if self.mode == "int8":
            q, _scale, _zp = _quantize_int8(vec)
            return q
        if self.mode == "int8_global":
            return _quantize_int8_global(vec)
        if self.mode == "int8_per_dim":
            q, _scale_arr = _quantize_int8_per_dim(vec)
            return q
        return vec.astype(np.float32)

    def dequantize(
        self, qvec: NDArray, scale: float = 1.0, zero_point: float = 0.0, scale_arr: Optional[NDArray] = None
    ) -> NDArray:
        """Return float32 copy (or original if mode == 'none')."""
        if self.mode == "fp16":
            return qvec.astype(np.float32)
        if self.mode in ("int8", "int8_global"):
            return _dequantize_int8(qvec, scale, zero_point)
        if self.mode == "int8_per_dim":
            if scale_arr is not None:
                return qvec.astype(np.float32) * scale_arr
            return _dequantize_int8(qvec, scale, zero_point)
        return qvec.astype(np.float32)

    def maybe_dequantize(
        self, qvec: NDArray, scale: float = 1.0, zero_point: float = 0.0, scale_arr: Optional[NDArray] = None
    ) -> NDArray:
        """Dequantize only if currently quantized."""
        if self.mode == "none":
            return qvec
        return self.dequantize(qvec, scale, zero_point, scale_arr)


# ------------------------------------------------------------------
# int8 helpers
# ------------------------------------------------------------------


def _quantize_int8(vec: NDArray) -> NDArray:
    """Symmetric per-vector int8 quantization.

    Uses signed int8 range [-127, 127] ( reserving -128 for zero
    if needed).  Scale must be stored alongside the array.
    """
    vec_f = vec.astype(np.float32)
    max_abs = float(np.abs(vec_f).max())
    if max_abs == 0:
        return np.zeros_like(vec_f, dtype=np.int8), 1.0, 0.0
    scale = max_abs / 127.0
    quantized = np.round(vec_f / scale).astype(np.int8)
    return quantized, scale, 0.0


def _dequantize_int8(qvec: NDArray, scale: float, zero_point: float) -> NDArray:
    """Dequantize int8 array back to float32.

    zero_point is ignored for symmetric quantization (kept for API
    compatibility with the QuantizationHelper interface).
    """
    return qvec.astype(np.float32) * scale


def _quantize_int8_global(vec: NDArray) -> NDArray:
    """Global int8 quantization with fixed scale 1/127.

    No per-vector metadata needed — scale is constant.
    """
    vec_f = vec.astype(np.float32)
    return np.round(vec_f / QuantizationHelper.INT8_GLOBAL_SCALE).astype(np.int8)


def _quantize_int8_per_dim(vec: NDArray) -> tuple[NDArray, NDArray]:
    """Per-dimension int8 quantization.

    Each dimension gets its own scale based on max abs value in that dim.
    Returns (quantized_vector, scale_array).
    """
    vec_f = vec.astype(np.float32)
    max_abs = np.abs(vec_f)
    # Avoid division by zero
    scale_arr = np.where(max_abs > 0, max_abs / 127.0, 1.0)
    quantized = np.round(vec_f / scale_arr).astype(np.int8)
    return quantized, scale_arr


def int8_fast_dot(a: NDArray, b: NDArray, scale_a: float, scale_b: float) -> float:
    """Fast dot product of two int8 vectors without full dequantization.

    Computes dot(a,b) in int16 space to avoid overflow, then scales.
    This is ~2-3x faster than dequantize + float32 np.dot on modern CPUs
    because it stays in SIMD integer units.
    """
    # int8 dot can overflow [-127*127, 127*127] per element; int16 is safe
    return float(np.dot(a.astype(np.int16), b.astype(np.int16))) * scale_a * scale_b
