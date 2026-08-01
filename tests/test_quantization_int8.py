"""Tests for int8 quantization."""

import numpy as np
import pytest

from rtmdk.memory.quantization import QuantizationHelper


class TestQuantizationHelper:
    def test_none_mode(self):
        qh = QuantizationHelper("none")
        assert qh.dtype == np.float32
        assert qh.itemsize == 4
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q = qh.quantize(vec)
        assert q.dtype == np.float32
        np.testing.assert_allclose(q, vec)

    def test_fp16_mode(self):
        qh = QuantizationHelper("fp16")
        assert qh.dtype == np.float16
        assert qh.itemsize == 2
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q = qh.quantize(vec)
        assert q.dtype == np.float16
        dq = qh.dequantize(q)
        np.testing.assert_allclose(dq, vec, atol=1e-3)

    def test_int8_mode_dtype(self):
        qh = QuantizationHelper("int8")
        assert qh.dtype == np.int8
        assert qh.itemsize == 1

    def test_int8_roundtrip(self):
        qh = QuantizationHelper("int8")
        vec = np.array([-1.5, 0.0, 2.5, 5.0], dtype=np.float32)
        q, scale, zp = qh.quantize_with_meta(vec)
        assert q.dtype == np.int8
        assert scale > 0
        dq = qh.dequantize(q, scale, zp)
        np.testing.assert_allclose(dq, vec, atol=0.05)

    def test_int8_itemsize_reduction(self):
        qh = QuantizationHelper("int8")
        vec = np.random.randn(256).astype(np.float32)
        q, _, _ = qh.quantize_with_meta(vec)
        assert q.nbytes == vec.nbytes // 4

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            QuantizationHelper("int4")

    def test_int8_all_same_values(self):
        qh = QuantizationHelper("int8")
        vec = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        q, scale, zp = qh.quantize_with_meta(vec)
        assert scale > 0
        assert zp == 0.0
        assert np.all(q == 127)  # symmetric: max value maps to 127

    def test_int8_negative_values(self):
        qh = QuantizationHelper("int8")
        vec = np.array([-10.0, -5.0, 0.0], dtype=np.float32)
        q, scale, zp = qh.quantize_with_meta(vec)
        dq = qh.dequantize(q, scale, zp)
        np.testing.assert_allclose(dq, vec, atol=0.1)

    def test_maybe_dequantize_none(self):
        qh = QuantizationHelper("none")
        vec = np.array([1.0, 2.0])
        assert qh.maybe_dequantize(vec) is vec

    def test_int8_global_mode(self):
        qh = QuantizationHelper("int8_global")
        assert qh.dtype == np.int8
        vec = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        q = qh.quantize(vec)
        assert q.dtype == np.int8
        dq = qh.dequantize(q, qh.INT8_GLOBAL_SCALE)
        np.testing.assert_allclose(dq, vec, atol=0.02)

    def test_int8_per_dim_mode(self):
        qh = QuantizationHelper("int8_per_dim")
        assert qh.dtype == np.int8
        vec = np.array([-1.0, 0.0, 5.0, 10.0], dtype=np.float32)
        q, scale, zp, scale_arr = qh.quantize_with_meta(vec)
        assert q.dtype == np.int8
        assert scale_arr is not None
        assert len(scale_arr) == len(vec)
        dq = qh.dequantize(q, scale_arr=scale_arr)
        np.testing.assert_allclose(dq, vec, atol=0.1)

    def test_memory_node_int8_property(self):
        from rtmdk.nodes import MemoryNode

        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        node = MemoryNode(
            id="test",
            latent_pos=vec,
            phase=0.0,
            amplitude=1.0,
            salience=1.0,
        )
        # Property returns float32
        assert node.latent_pos.dtype == np.float32
        np.testing.assert_allclose(node.latent_pos, vec)

    def test_memory_node_int8_dequantize(self):
        from rtmdk.nodes import MemoryNode

        vec = np.array([0.2, 0.5, 1.0], dtype=np.float32)
        q = (vec / QuantizationHelper.INT8_GLOBAL_SCALE).astype(np.int8)
        node = MemoryNode(
            id="test",
            latent_pos=q,
            phase=0.0,
            amplitude=1.0,
            salience=1.0,
            latent_scale=QuantizationHelper.INT8_GLOBAL_SCALE,
            latent_zero_point=0.0,
        )
        dq = node.latent_pos
        assert dq.dtype == np.float32
        np.testing.assert_allclose(dq, vec, atol=0.1)
