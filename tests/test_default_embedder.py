"""Batteries-included quickstart: RTMDKMemory works without an embedder."""

import numpy as np
import pytest

from rtmdk import RTMDKConfig, RTMDKMemory
from rtmdk.memory.default_embedder import create_default_embedder


class TestDefaultEmbedder:
    def test_embedder_shape_and_dtype(self):
        embed = create_default_embedder(dim=64)
        vec = embed("hello world")
        assert vec.shape == (64,)
        assert vec.dtype == np.float32

    def test_deterministic(self):
        e1 = create_default_embedder(dim=32)
        e2 = create_default_embedder(dim=32)
        np.testing.assert_array_equal(e1("same text"), e2("same text"))

    def test_lexical_similarity(self):
        embed = create_default_embedder(dim=64)
        a = embed("coffee in the morning")
        b = embed("morning coffee")
        c = embed("quantum chromodynamics")
        assert a @ b > a @ c


class TestQuickstartAPI:
    @pytest.fixture()
    def memory(self):
        cfg = RTMDKConfig(latent_dim=32, embedding_dim=32, use_hnsw=False)
        return RTMDKMemory(config=cfg)  # no embedder — batteries included

    def test_construct_without_embedder(self, memory):
        assert memory.embedder is not None

    def test_add_and_query_three_lines(self, memory):
        memory.add("RTMDK is a resonance-topological memory for LLMs")
        memory.add("SOT tokenizer learns byte-level embeddings")
        results = memory.query("resonance memory", top_k=2)
        assert results, "query returned nothing"
        top_text = results[0][2].content.get("text", "")
        assert "RTMDK" in top_text or "resonance" in top_text

    def test_explicit_embedder_still_wins(self):
        cfg = RTMDKConfig(latent_dim=16, embedding_dim=16, use_hnsw=False)
        sentinel = lambda t: np.ones(16, dtype=np.float32)  # noqa: E731
        mem = RTMDKMemory(config=cfg, embedder=sentinel)
        # embedder may be wrapped (circuit breaker) but must BEHAVE like sentinel
        np.testing.assert_array_equal(mem.embedder("probe"), np.ones(16, dtype=np.float32))
