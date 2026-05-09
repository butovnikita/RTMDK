"""Tests for architectural backlog modules:
- EngramEmbeddingCache (hot/warm/cold tiers)
- Observability / Telemetry (latency histograms, cache ratios)
- DistributedLock (file backend)
- RAG Quality (query decomposition, sentence reranking, feedback loop)
"""
import os
import tempfile
import threading
import time
import numpy as np
import pytest


class TestEngramEmbeddingCache:
    def test_basic_add_get(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=2, max_warm=2)
        emb = np.array([1.0, 2.0, 3.0])
        cache.add("n1", emb)
        got = cache.get("n1")
        assert got is not None
        np.testing.assert_array_equal(got, emb)

    def test_put_alias(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=2, max_warm=2)
        cache.put("n1", np.array([1.0]))
        assert cache.get("n1") is not None

    def test_hot_eviction_moves_to_warm(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=2, max_warm=2)
        cache.add("n1", np.array([1.0]))
        cache.add("n2", np.array([2.0]))
        cache.add("n3", np.array([3.0]))  # evicts n1 to warm
        assert cache.get("n1") is not None
        assert cache.get("n2") is not None
        assert cache.get("n3") is not None

    def test_warm_eviction_drops_to_cold(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=1, max_warm=1)
        cache.add("n1", np.array([1.0]))
        cache.add("n2", np.array([2.0]))  # n1 warm
        cache.add("n3", np.array([3.0]))  # n2 warm, n1 cold (dropped)
        assert cache.get("n1") is None
        assert cache.get("n2") is not None
        assert cache.get("n3") is not None

    def test_warm_promotion_to_hot(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=1, max_warm=1)
        cache.add("n1", np.array([1.0]))
        cache.add("n2", np.array([2.0]))  # n1 warm
        assert "n1" not in cache._hot
        assert "n1" in cache._warm
        cache.get("n1")  # promotes to hot
        assert "n1" in cache._hot
        assert "n1" not in cache._warm

    def test_thread_safety(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=100, max_warm=100)
        errors = []

        def worker(i):
            try:
                cache.add(f"n{i}", np.array([float(i)]))
                time.sleep(0.001)
                got = cache.get(f"n{i}")
                if got is None or got[0] != float(i):
                    errors.append(f"mismatch {i}")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_get_all(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=10, max_warm=10)
        cache.add("n1", np.array([1.0]))
        cache.add("n2", np.array([2.0]))
        all_embs = cache.get_all()
        assert len(all_embs) == 2
        assert "n1" in all_embs
        assert "n2" in all_embs

    def test_remove(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=10, max_warm=10)
        cache.add("n1", np.array([1.0]))
        cache.remove("n1")
        assert cache.get("n1") is None
        assert "n1" not in cache

    def test_len(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=10, max_warm=10)
        assert len(cache) == 0
        cache.add("n1", np.array([1.0]))
        assert len(cache) == 1

    def test_save_load(self):
        from rtmdk.memory.engram_cache import EngramEmbeddingCache
        cache = EngramEmbeddingCache(max_hot=10, max_warm=10)
        cache.add("n1", np.array([1.0, 2.0]))
        cache.add("n2", np.array([3.0, 4.0]))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cache.npz")
            cache.save(path)
            cache2 = EngramEmbeddingCache(max_hot=10, max_warm=10)
            cache2.load(path)
            assert len(cache2) == 2
            np.testing.assert_array_equal(cache2.get("n1"), np.array([1.0, 2.0]))
            np.testing.assert_array_equal(cache2.get("n2"), np.array([3.0, 4.0]))


class TestObservability:
    def test_record_latency_percentiles(self):
        from rtmdk.memory.observability import MemoryMetrics
        metrics = MemoryMetrics()
        for i in range(1, 101):
            metrics.record_query(float(i), cache_hit=False)
        p = metrics.query_latency.percentiles()
        assert 50 <= p["p50"] <= 51
        assert 95 <= p["p95"] <= 96
        assert 99 <= p["p99"] <= 100

    def test_cache_hit_ratio(self):
        from rtmdk.memory.observability import MemoryMetrics
        metrics = MemoryMetrics()
        for _ in range(3):
            metrics.record_query(1.0, cache_hit=True)
        for _ in range(7):
            metrics.record_query(1.0, cache_hit=False)
        assert abs(metrics.cache_hit_ratio() - 0.3) < 0.01

    def test_alert_rule(self):
        from rtmdk.memory.observability import AlertRule
        rule = AlertRule("high_latency", "query_p99", threshold=50.0)
        assert not rule.check(30.0, time.time())
        assert rule.check(60.0, time.time())
        # Cooldown should prevent immediate re-fire
        assert not rule.check(60.0, time.time())

    def test_check_alerts(self):
        from rtmdk.memory.observability import MemoryMetrics, AlertRule
        metrics = MemoryMetrics()
        metrics.add_alert_rule(AlertRule("high_latency", "query_p99", threshold=50.0))
        metrics.record_query(100.0, cache_hit=False)
        alerts = metrics.check_alerts()
        assert len(alerts) >= 1
        assert "high_latency" in alerts[0]

    def test_prometheus_export(self):
        from rtmdk.memory.observability import MemoryMetrics
        metrics = MemoryMetrics()
        metrics.record_query(10.0, cache_hit=False)
        prom = metrics.to_prometheus()
        assert "rtmdk_query_latency_p50" in prom
        assert "rtmdk_cache_hit_ratio" in prom

    def test_flush_to_file(self):
        from rtmdk.memory.observability import MemoryMetrics
        metrics = MemoryMetrics()
        metrics.record_query(10.0, cache_hit=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.json")
            metrics.flush_to_file(path)
            assert os.path.exists(path)
            import json
            with open(path) as f:
                data = json.load(f)
            assert data["cache_hit_ratio"] == 1.0


class TestDistributedLock:
    def test_file_lock_acquire_release(self):
        from rtmdk.memory.distributed_lock import DistributedLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = DistributedLock(os.path.join(tmpdir, "lock"))
            assert lock.acquire(blocking=False)
            lock.release()

    def test_file_lock_blocks_second_acquirer(self):
        from rtmdk.memory.distributed_lock import DistributedLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock1 = DistributedLock(os.path.join(tmpdir, "lock"))
            lock2 = DistributedLock(os.path.join(tmpdir, "lock"))
            assert lock1.acquire(blocking=False)
            assert not lock2.acquire(blocking=False)
            lock1.release()

    def test_file_lock_timeout(self):
        from rtmdk.memory.distributed_lock import DistributedLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock1 = DistributedLock(os.path.join(tmpdir, "lock"))
            lock2 = DistributedLock(os.path.join(tmpdir, "lock"), timeout=0.1)
            assert lock1.acquire(blocking=False)
            assert not lock2.acquire(blocking=True)
            lock1.release()

    def test_file_lock_thread_safety(self):
        from rtmdk.memory.distributed_lock import DistributedLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = DistributedLock(os.path.join(tmpdir, "lock"))
            acquired = []

            def worker():
                if lock.acquire(blocking=True):
                    time.sleep(0.05)
                    lock.release()
                    acquired.append(True)

            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert len(acquired) == 5

    def test_redis_fallback_when_unavailable(self):
        from rtmdk.memory.distributed_lock import DistributedLock
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = DistributedLock(
                os.path.join(tmpdir, "lock"),
                backend="redis",
                redis_url="redis://localhost:9999/0",
            )
            # Should fall back to file backend
            assert lock.backend == "file"
            assert lock.acquire(blocking=False)
            lock.release()


class TestRAGQuality:
    def test_query_decomposer_splits_multi_hop(self):
        from rtmdk.memory.rag_quality import QueryDecomposer
        dec = QueryDecomposer()
        sub = dec.decompose("What is X and who invented Y")
        assert len(sub) >= 1
        assert any("X" in s for s in sub)

    def test_query_decomposer_single_query_unchanged(self):
        from rtmdk.memory.rag_quality import QueryDecomposer
        dec = QueryDecomposer()
        sub = dec.decompose("simple query")
        assert sub == ["simple query"]

    def test_query_decomposer_llm_fallback(self):
        from rtmdk.memory.rag_quality import QueryDecomposer

        class BadLLM:
            def complete(self, prompt, max_tokens=200):
                raise RuntimeError("fail")

        dec = QueryDecomposer(llm_client=BadLLM())
        sub = dec.decompose("What is X and who invented Y")
        assert len(sub) >= 1

    def test_sentence_reranker_returns_top_k(self):
        from rtmdk.memory.rag_quality import SentenceReranker

        class MockEmbedder:
            def __call__(self, text):
                return np.ones(4, dtype=np.float32)

        class MockNode:
            def __init__(self, text):
                self.content = {"text": text}

        reranker = SentenceReranker(MockEmbedder())
        results = [
            (f"n{i}", float(i), MockNode("Sentence one. Sentence two."))
            for i in range(10)
        ]
        reranked = reranker.rerank("query", results, top_k=3)
        assert len(reranked) == 3

    def test_feedback_loop_records(self):
        from rtmdk.memory.rag_quality import FeedbackLoop
        class MockSIF:
            word_embeddings = {0: np.ones(4), 1: np.ones(4)}
            def _embed_sentence_raw(self, tokens):
                return np.ones(4)
        class MockEmbedder:
            _vocab = {"q1": 0, "node": 1}
            _embedder = MockSIF()
            def _word_tokenize(self, text):
                return text.lower().split()
        fl = FeedbackLoop(MockEmbedder())
        assert fl.add_feedback("q1", "node", True)
        assert fl.feedback_count == 1

    def test_feedback_loop_persistence(self):
        from rtmdk.memory.rag_quality import FeedbackLoop
        class MockSIF:
            word_embeddings = {0: np.ones(4), 1: np.ones(4)}
            def _embed_sentence_raw(self, tokens):
                return np.ones(4)
        class MockEmbedder:
            _vocab = {"q1": 0, "node": 1}
            _embedder = MockSIF()
            def _word_tokenize(self, text):
                return text.lower().split()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "feedback.json")
            fl = FeedbackLoop(MockEmbedder(), persist_path=path)
            fl.add_feedback("q1", "node", True)
            fl.add_feedback("q1", "node", False)
            fl._flush()
            assert os.path.exists(path)
            fl2 = FeedbackLoop(MockEmbedder(), persist_path=path)
            fl2.load()
            assert fl2.feedback_count == 2
