"""Unit tests for QuantizationHelper."""

import numpy as np
import pytest

from rtmdk.memory.quantization import QuantizationHelper, _quantize_int8, _dequantize_int8


class TestQuantizationHelper:
    def test_none_mode_dtype(self):
        q = QuantizationHelper("none")
        assert q.dtype == np.float32
        assert q.itemsize == 4

    def test_fp16_mode_dtype(self):
        q = QuantizationHelper("fp16")
        assert q.dtype == np.float16
        assert q.itemsize == 2

    def test_int8_mode_dtype(self):
        q = QuantizationHelper("int8")
        assert q.dtype == np.int8
        assert q.itemsize == 1

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported quantization mode"):
            QuantizationHelper("int4")

    def test_quantize_none(self):
        q = QuantizationHelper("none")
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = q.quantize(vec)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, vec)

    def test_quantize_fp16(self):
        q = QuantizationHelper("fp16")
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = q.quantize(vec)
        assert out.dtype == np.float16
        np.testing.assert_array_almost_equal(out.astype(np.float32), vec, decimal=3)

    def test_quantize_int8_returns_ndarray(self):
        """quantize() must return NDArray, not a tuple (regression guard)."""
        q = QuantizationHelper("int8")
        vec = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        out = q.quantize(vec)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int8

    def test_quantize_with_meta_int8(self):
        q = QuantizationHelper("int8")
        vec = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        qvec, scale, zp = q.quantize_with_meta(vec)
        assert qvec.dtype == np.int8
        assert scale > 0
        assert zp == 0.0

    def test_dequantize_int8_roundtrip(self):
        q = QuantizationHelper("int8")
        vec = np.array([1.0, -2.0, 3.0, 0.5, -0.25], dtype=np.float32)
        qvec, scale, zp = q.quantize_with_meta(vec)
        recon = q.dequantize(qvec, scale, zp)
        assert recon.dtype == np.float32
        # Symmetric int8 roundtrip should be close for typical vectors
        np.testing.assert_array_almost_equal(recon, vec, decimal=2)

    def test_maybe_dequantize_none(self):
        q = QuantizationHelper("none")
        vec = np.array([1.0, 2.0], dtype=np.float32)
        out = q.maybe_dequantize(vec)
        assert out is vec  # same object, no copy

    def test_zero_vector_int8(self):
        """Zero vector should quantize to all zeros with scale=1.0."""
        vec = np.zeros(10, dtype=np.float32)
        qvec, scale, zp = _quantize_int8(vec)
        assert qvec.dtype == np.int8
        assert np.all(qvec == 0)
        assert scale == 1.0
        assert zp == 0.0

    def test_int8_recall_high(self):
        """int8 quantization should preserve cosine similarity ~98%."""
        rng = np.random.RandomState(42)
        vec = rng.randn(384).astype(np.float32)
        qvec, scale, zp = _quantize_int8(vec)
        recon = _dequantize_int8(qvec, scale, zp)
        # Cosine similarity should be very high
        sim = np.dot(vec, recon) / (np.linalg.norm(vec) * np.linalg.norm(recon))
        assert sim >= 0.98
