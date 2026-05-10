"""End-to-end test for int8 quantization mode."""

import numpy as np

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory


def _embedder(text: str):
    rng = np.random.RandomState(hash(text) % 2**31)
    return rng.randn(64).astype(np.float32)


class TestQuantizationE2E:
    def test_int8_add_and_query(self):
        cfg = RTMDKConfig(latent_dim=64, quantization="int8", use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)

        for i in range(10):
            emb = np.random.randn(64).astype(np.float32)
            mem.add_node(
                content={"text": f"node {i}", "topic": "test"},
                embedding=emb,
            )

        assert len(mem.field.nodes) == 10
        assert mem.field._quant.mode == "int8"

        q = np.random.randn(64).astype(np.float32)
        results = mem.query_with_confidence(query="test query", embedding=q, top_k=3)
        assert isinstance(results, dict)

    def test_fp16_add_and_query(self):
        cfg = RTMDKConfig(latent_dim=64, quantization="fp16", use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)

        for i in range(5):
            emb = np.random.randn(64).astype(np.float32)
            mem.add_node(
                content={"text": f"node {i}", "topic": "test"},
                embedding=emb,
            )

        assert mem.field._quant.mode == "fp16"
        q = np.random.randn(64).astype(np.float32)
        results = mem.query_with_confidence(query="test", embedding=q, top_k=2)
        assert isinstance(results, dict)

    def test_none_quantization_default(self):
        cfg = RTMDKConfig(latent_dim=64, quantization="none", use_hnsw=False)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)

        for i in range(5):
            emb = np.random.randn(64).astype(np.float32)
            mem.add_node(
                content={"text": f"node {i}", "topic": "test"},
                embedding=emb,
            )

        assert mem.field._quant.mode == "none"
        q = np.random.randn(64).astype(np.float32)
        results = mem.query_with_confidence(query="test", embedding=q, top_k=2)
        assert isinstance(results, dict)
