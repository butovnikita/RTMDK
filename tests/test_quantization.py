"""tests/test_quantization.py — Quantization integration tests."""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.quantization import QuantizationHelper

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"


class TestQuantizationHelper:
    def test_none(self):
        q = QuantizationHelper("none")
        v = np.array([1.0, -0.5, 0.0], dtype=np.float32)
        assert q.quantize(v).dtype == np.float32
        assert np.allclose(q.dequantize(q.quantize(v)), v)

    def test_fp16(self):
        q = QuantizationHelper("fp16")
        v = np.array([1.0, -0.5, 0.0], dtype=np.float32)
        qv = q.quantize(v)
        assert qv.dtype == np.float16
        dv = q.dequantize(qv)
        assert dv.dtype == np.float32
        assert np.allclose(dv, v, atol=1e-3)

    def test_fp16_top1_preserved(self):
        """fp16 quantization preserves top-1 on random unit vectors."""
        np.random.seed(42)
        dim = 256
        n = 1000
        pos = np.random.randn(n, dim).astype(np.float32)
        pos /= np.linalg.norm(pos, axis=1, keepdims=True)
        q = QuantizationHelper("fp16")
        qpos = np.stack([q.quantize(p) for p in pos])
        query = np.random.randn(dim).astype(np.float32)
        query /= np.linalg.norm(query)
        top1_orig = int(np.argmax(pos @ query))
        # Dequantize before dot product (simulates field query path)
        dpos = q.dequantize(qpos)
        top1_q = int(np.argmax(dpos @ query))
        assert top1_orig == top1_q


class TestFieldQuantization:
    def test_fp16_field_reduces_memory(self):
        cfg = RTMDKConfig(
            latent_dim=64, top_k=5, min_response=0.001,
            decay_rate=0.999, use_hnsw=False, learn_projection=False,
            bm25_fallback=False, enable_async=False,
            resonance_kernel="cosine", phase_coupling=0.0,
            quantization="fp16",
        )
        field = RTMDKField(cfg)
        pos = np.random.randn(100, 64).astype(np.float32)
        pos /= np.linalg.norm(pos, axis=1, keepdims=True)
        for i in range(100):
            field.add_node(pos[i], content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
        # All node latent positions should be fp16
        for nid in field.node_index:
            assert field.nodes[nid].latent_pos.dtype == np.float16
        # Cache should also be fp16
        field.query(pos[0], top_k=5)  # trigger cache build
        assert field._cached_positions.dtype == np.float16

    def test_fp16_recall_vs_baseline(self):
        np.random.seed(42)
        dim = 64
        n_nodes = 500
        n_queries = 50
        positions = np.random.randn(n_nodes, dim).astype(np.float32)
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        base_cfg = RTMDKConfig(
            latent_dim=dim, top_k=5, min_response=0.001,
            decay_rate=0.999, use_hnsw=False, learn_projection=False,
            bm25_fallback=False, enable_async=False,
            resonance_kernel="cosine", phase_coupling=0.0,
            quantization="none",
        )
        base_field = RTMDKField(base_cfg)
        for i in range(n_nodes):
            base_field.add_node(positions[i], content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
            base_field.nodes[f"n{i}"].amplitude = 1.0
            base_field.nodes[f"n{i}"].salience = 1.0

        q_cfg = RTMDKConfig(
            latent_dim=dim, top_k=5, min_response=0.001,
            decay_rate=0.999, use_hnsw=False, learn_projection=False,
            bm25_fallback=False, enable_async=False,
            resonance_kernel="cosine", phase_coupling=0.0,
            quantization="fp16",
        )
        q_field = RTMDKField(q_cfg)
        for i in range(n_nodes):
            q_field.add_node(positions[i], content={"id": i}, phase=0.0, node_id=f"n{i}", skip_projection=True)
            q_field.nodes[f"n{i}"].amplitude = 1.0
            q_field.nodes[f"n{i}"].salience = 1.0

        hits = 0
        for q in queries:
            r_base = base_field.query(q, top_k=1)
            r_q = q_field.query(q, top_k=1)
            if r_base and r_q and r_base[0][0] == r_q[0][0]:
                hits += 1
        recall = hits / n_queries
        assert recall >= 0.98, f"fp16 R@1 recall {recall} too low"
