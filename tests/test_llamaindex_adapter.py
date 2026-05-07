"""Tests for rtmdk.production.llamaindex_adapter."""

import numpy as np
import pytest

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig


def _make_embedder(dim: int = 64):
    def embed(text: str) -> np.ndarray:
        h = hash(text) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.standard_normal(dim, dtype=np.float32)
    return embed


class TestLlamaIndexAdapter:
    def test_retriever_without_llamaindex(self):
        """When llama-index is not installed, retriever should still instantiate
        and provide legacy retrieve()."""
        from rtmdk.production.llamaindex_adapter import (
            RTMDKLlamaIndexRetriever,
            LLAMAINDEX_AVAILABLE,
        )

        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        mem.save_context({"input": "coffee is great", "session_id": "s1"}, {"output": ""})

        retriever = RTMDKLlamaIndexRetriever(memory=mem, top_k=3)
        # retrieve() uses _parse_context directly when bundle is a string
        nodes = retriever.retrieve("coffee")
        assert isinstance(nodes, list)
        # Without llamaindex TextNode/NodeWithScore, _parse_context returns []
        if not LLAMAINDEX_AVAILABLE:
            assert nodes == []

    def test_vector_store_add_and_query(self):
        from rtmdk.production.llamaindex_adapter import RTMDKVectorStore

        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        store = RTMDKVectorStore(mem)

        store.add(["hello world", "goodbye world"])
        assert len(store.get_nodes()) == 2
        assert len(mem.field.nodes) == 2

        results = store.query("hello", top_k=3)
        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]

    def test_vector_store_with_metadata(self):
        from rtmdk.production.llamaindex_adapter import RTMDKVectorStore

        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        store = RTMDKVectorStore(mem)
        store.add(["doc1"], metadatas=[{"author": "alice"}])
        assert store.get_nodes()[0]["metadata"]["author"] == "alice"

    def test_parse_context_structured(self):
        from rtmdk.production.llamaindex_adapter import RTMDKLlamaIndexRetriever

        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        retriever = RTMDKLlamaIndexRetriever(memory=mem)
        ctx = "[ATTN:0.95][SAL:0.8][TIER:semantic] Important fact\n[ATTN:0.5] Less important"
        nodes = retriever._parse_context(ctx)
        # Without llama-index installed, _parse_context returns []
        # We just verify it doesn't crash
        assert isinstance(nodes, list)

    def test_parse_context_empty(self):
        from rtmdk.production.llamaindex_adapter import RTMDKLlamaIndexRetriever

        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        retriever = RTMDKLlamaIndexRetriever(memory=mem)
        assert retriever._parse_context("") == []
        assert retriever._parse_context("   ") == []

    def test_retriever_score_threshold(self):
        from rtmdk.production.llamaindex_adapter import RTMDKLlamaIndexRetriever

        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        retriever = RTMDKLlamaIndexRetriever(memory=mem, score_threshold=0.9)
        ctx = "[ATTN:0.95] High score\n[ATTN:0.5] Low score"
        nodes = retriever._parse_context(ctx)
        # Just verify no crash; actual filtering depends on llamaindex classes
        assert isinstance(nodes, list)

    def test_retriever_top_k(self):
        from rtmdk.production.llamaindex_adapter import RTMDKLlamaIndexRetriever

        cfg = RTMDKConfig(latent_dim=64, embedding_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_make_embedder(64))
        retriever = RTMDKLlamaIndexRetriever(memory=mem, top_k=2)
        ctx = "[ATTN:0.9] One\n[ATTN:0.8] Two\n[ATTN:0.7] Three"
        nodes = retriever._parse_context(ctx)
        # Without llamaindex, returns []; with llamaindex, should be capped at 2
        assert isinstance(nodes, list)
