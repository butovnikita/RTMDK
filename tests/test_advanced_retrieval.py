"""Tests for rtmdk.production.advanced_retrieval."""

import numpy as np
import pytest
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
from rtmdk.production.advanced_retrieval import (
    HybridRetriever,
    ConfidenceAwareFallback,
    QueryExpander,
    AdaptiveDepthRetriever,
    TemporalDecayLearner,
    CausalAugmentedRetriever,
    MetaRetrievalController,
    AdvancedRTMDKRetriever,
)
from rtmdk.production.bm25_fallback import BM25FallbackRetriever


def _embed(text: str) -> np.ndarray:
    return np.random.randn(768).astype(np.float32)


def _make_mem():
    cfg = RTMDKConfig(latent_dim=64)
    return RTMDKMemory(config=cfg, embedder=_embed)


def _make_bm25():
    return BM25FallbackRetriever()


class TestHybridRetriever:
    def test_retrieve_basic(self):
        mem = _make_mem()
        bm25 = _make_bm25()
        retriever = HybridRetriever(mem, bm25)
        mem.save_context({"input": "coffee", "session_id": "s1"}, {"output": ""})
        emb = _embed("coffee")
        results = retriever.retrieve("coffee", emb, top_k=3)
        assert isinstance(results, list)

    def test_add_embedding(self):
        mem = _make_mem()
        bm25 = _make_bm25()
        retriever = HybridRetriever(mem, bm25)
        emb = np.random.randn(64).astype(np.float32)
        retriever.add_embedding("n1", emb)
        assert "n1" in retriever._embeddings_cache


class TestConfidenceAwareFallback:
    def test_confident_result(self):
        mem = _make_mem()
        bm25 = _make_bm25()
        hybrid = HybridRetriever(mem, bm25)
        cf = ConfidenceAwareFallback(hybrid, bm25)
        mem.save_context({"input": "coffee", "session_id": "s1"}, {"output": ""})
        emb = _embed("coffee")
        results, status = cf.retrieve("coffee", emb, top_k=3)
        assert status in ("confident", "fallback", "unknown")


class TestQueryExpander:
    def test_expand_no_context(self):
        mem = _make_mem()
        expander = QueryExpander(mem)
        assert expander.expand("test query") == "test query"

    def test_extract_significant_words(self):
        text = "The brown fox jumps over the lazy dog"
        words = QueryExpander._extract_significant_words(text)
        assert "the" not in words  # stopword
        assert "brown" in words


class TestAdaptiveDepthRetriever:
    def test_adaptive_depth(self):
        mem = _make_mem()
        bm25 = _make_bm25()
        hybrid = HybridRetriever(mem, bm25)
        adaptive = AdaptiveDepthRetriever(hybrid)
        mem.save_context({"input": "a", "session_id": "s1"}, {"output": ""})
        mem.save_context({"input": "b", "session_id": "s1"}, {"output": ""})
        emb = _embed("a")
        results = adaptive.retrieve("a", emb, top_k=3)
        assert isinstance(results, list)


class TestTemporalDecayLearner:
    def test_feedback_and_get_decay(self):
        learner = TemporalDecayLearner()
        learner.apply_feedback("n1", 0.8)
        decay = learner.get_decay_rate("n1")
        assert learner.min_decay <= decay <= learner.max_decay

    def test_apply_to_node(self):
        from rtmdk.nodes import MemoryNode
        node = MemoryNode.__new__(MemoryNode)
        node.salience = 1.0
        learner = TemporalDecayLearner()
        learner.apply_to_node(node, "n1")
        assert node.salience < 1.0

    def test_stats(self):
        learner = TemporalDecayLearner()
        stats = learner.stats
        assert "avg_decay" in stats


class TestCausalAugmentedRetriever:
    def test_retrieve(self):
        mem = _make_mem()
        retriever = CausalAugmentedRetriever(mem)
        mem.save_context({"input": "a", "session_id": "s1"}, {"output": ""})
        emb = _embed("a")
        results = retriever.retrieve("a", emb, top_k=3)
        assert isinstance(results, list)


class TestMetaRetrievalController:
    def test_classify_query(self):
        strategies = {}
        ctrl = MetaRetrievalController(strategies)
        qtype = ctrl.classify_query("What is the capital of France?")
        assert qtype == "factual"

    def test_classify_vague(self):
        strategies = {}
        ctrl = MetaRetrievalController(strategies)
        qtype = ctrl.classify_query("Remember that thing we discussed?")
        assert qtype == "vague"

    def test_stats(self):
        strategies = {"factual": None}
        ctrl = MetaRetrievalController(strategies)
        assert ctrl.stats["total_queries"] == 0


class TestAdvancedRTMDKRetriever:
    def test_retrieve_basic(self):
        mem = _make_mem()
        bm25 = _make_bm25()
        adv = AdvancedRTMDKRetriever(mem, bm25)
        for i in range(10):
            mem.save_context({"input": f"hello {i}", "session_id": "s1"}, {"output": ""})
        emb = _embed("hello")
        results, qtype = adv.retrieve("hello", emb, top_k=3)
        assert isinstance(results, list)
        assert isinstance(qtype, str)

    def test_apply_feedback(self):
        mem = _make_mem()
        bm25 = _make_bm25()
        adv = AdvancedRTMDKRetriever(mem, bm25)
        adv.apply_feedback("q", ["n1"], 0.8)

    def test_get_stats(self):
        mem = _make_mem()
        bm25 = _make_bm25()
        adv = AdvancedRTMDKRetriever(mem, bm25)
        stats = adv.get_stats()
        assert "hybrid_enabled" in stats
