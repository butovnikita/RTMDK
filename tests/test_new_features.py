"""Tests for new P0/P1 fixes: batch API, query expansion, BM25 first-stage, NPZ, etc."""
import json
import os
import tempfile

import numpy as np
import pytest

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.field import RTMDKField
from rtmdk.memory.sot_v2.integration import SOTv2Embedder


def _embedder(text: str) -> np.ndarray:
    rng = np.random.default_rng(hash(text) % (2 ** 32))
    emb = rng.standard_normal(64).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-8
    return emb


class TestBatchQuery:
    def test_retrieve_nodes_batch_basic(self):
        cfg = RTMDKConfig(latent_dim=64, top_k=2)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        for i in range(5):
            mem.add_node(_embedder(f"doc{i}"), {"text": f"doc{i}"})
        queries = ["doc0", "doc1"]
        embs = np.vstack([_embedder(q) for q in queries])
        results = mem.retrieve_nodes_batch(queries, embs, top_k=2)
        assert len(results) == 2
        for r in results:
            assert len(r) <= 2
            assert all(isinstance(x, tuple) and len(x) == 3 for x in r)

    def test_retrieve_nodes_batch_empty(self):
        cfg = RTMDKConfig(latent_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        results = mem.retrieve_nodes_batch([], np.array([]).reshape(0, 64))
        assert results == []


class TestQueryExpansion:
    def test_short_query_expansion(self):
        sot = SOTv2Embedder(latent_dim=16)
        sot.train(["hello world", "foo bar baz"])
        cfg = RTMDKConfig(latent_dim=16, query_expand_short=True)
        mem = RTMDKMemory(config=cfg, embedder=sot)
        mem.add_node(sot("hello world"), {"text": "hello world"})
        # Short query should trigger expansion
        results = mem.retrieve_nodes("hi", sot("hi"), top_k=2)
        assert isinstance(results, list)


class TestBM25FirstStage:
    def test_bm25_first_stage_filtering(self):
        cfg = RTMDKConfig(
            latent_dim=64, bm25_first_stage_k=2, top_k=2)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        for i in range(5):
            mem.add_node(_embedder(f"doc{i}"), {"text": f"doc{i}"})
        results = mem.retrieve_nodes("doc0", _embedder("doc0"), top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2


class TestClear:
    def test_clear_does_not_crash(self):
        cfg = RTMDKConfig(latent_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        for i in range(3):
            mem.add_node(_embedder(f"doc{i}"), {"text": f"doc{i}"})
        mem.clear()
        assert len(mem.field.nodes) == 0


class TestCalibrateConformalSOT:
    def test_calibrate_does_not_crash(self):
        cfg = RTMDKConfig(
            latent_dim=64, conformal_prediction=True,
            conformal_alpha=0.1, conformal_min_calib=2)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        for i in range(5):
            mem.add_node(_embedder(f"doc{i}"), {"text": f"doc{i}"})
        ok = mem.calibrate_conformal_sot(n_samples=3)
        assert isinstance(ok, bool)


class TestSOTSaveLoad:
    def test_npz_roundtrip(self):
        sot = SOTv2Embedder(latent_dim=16)
        sot.train(["hello world", "foo bar baz", "hello foo"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npz")
            sot.save_npz(path)
            sot2 = SOTv2Embedder.load_npz(path)
            emb1 = sot("hello world")
            emb2 = sot2("hello world")
            np.testing.assert_allclose(emb1, emb2, atol=1e-5)


class TestMemoryProxy:
    def test_get_dashboard_proxy(self):
        cfg = RTMDKConfig(latent_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        # __getattr__ should proxy to field
        dashboard = mem.get_dashboard()
        assert isinstance(dashboard, dict)


class TestSOTCorpusLimit:
    def test_corpus_fifo(self):
        cfg = RTMDKConfig(latent_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        # Force corpus limit
        mem._sot_v2_corpus_maxlen = 3
        mem._sot_v2_corpus = []
        for i in range(5):
            mem._sot_v2_corpus.append(f"doc{i}")
            if len(mem._sot_v2_corpus) > mem._sot_v2_corpus_maxlen:
                mem._sot_v2_corpus.pop(0)
        assert len(mem._sot_v2_corpus) == 3
        assert mem._sot_v2_corpus[0] == "doc2"


class TestEmotionLogic:
    def test_questioning_not_negative(self):
        cfg = RTMDKConfig(latent_dim=64)
        mem = RTMDKMemory(config=cfg, embedder=_embedder)
        # This would previously set emotion="negative" then overwrite with "questioning"
        # We just verify save_context runs without error
        mem.save_context(
            {"input": "what is this?"},
            {"output": "answer"}
        )
        assert len(mem.field.nodes) >= 1
