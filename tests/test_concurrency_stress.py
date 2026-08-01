"""Concurrency stress tests for RTMDK core operations.

Spawns multiple threads performing simultaneous add_node and query
operations to detect race conditions in field-level data structures.
"""

import threading
import time

import numpy as np

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKField


def _make_field():
    cfg = RTMDKConfig(latent_dim=16, use_hnsw=False, rate_limit_nodes_per_sec=0)
    return RTMDKField(cfg)


class TestConcurrentAddNode:
    def test_concurrent_add_node_consistency(self):
        """Many threads adding nodes should not crash or lose data."""
        field = _make_field()
        num_threads = 8
        nodes_per_thread = 50
        errors = []

        def worker(tid):
            try:
                for i in range(nodes_per_thread):
                    emb = np.random.randn(16).astype(np.float32)
                    field.add_node(
                        embedding=emb,
                        content={"text": f"thread{tid}_node{i}"},
                        node_id=f"t{tid}_n{i}",
                    )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Exceptions during concurrent add: {errors}"
        assert len(field.nodes) == num_threads * nodes_per_thread

    def test_concurrent_add_node_with_cache_rebuild(self):
        """Concurrent adds while cache rebuilds should not crash."""
        field = _make_field()
        errors = []

        def adder():
            try:
                for i in range(100):
                    emb = np.random.randn(16).astype(np.float32)
                    field.add_node(embedding=emb, content={"c": i}, node_id=f"n{i}")
            except Exception as exc:
                errors.append(exc)

        def querier():
            try:
                for _ in range(50):
                    qemb = np.random.randn(16).astype(np.float32)
                    field.query(qemb, top_k=3)
                    time.sleep(0.005)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=querier)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not errors, f"Exceptions: {errors}"
        assert len(field.nodes) == 100


class TestConcurrentQuery:
    def test_concurrent_query_during_ingest(self):
        """Queries during heavy ingestion should not crash."""
        field = _make_field()
        # Pre-populate
        for i in range(200):
            emb = np.random.randn(16).astype(np.float32)
            field.add_node(embedding=emb, content={"text": f"doc{i}"}, node_id=f"n{i}")

        errors = []
        query_results = []

        def adder():
            try:
                for i in range(200, 400):
                    emb = np.random.randn(16).astype(np.float32)
                    field.add_node(embedding=emb, content={"text": f"doc{i}"}, node_id=f"n{i}")
            except Exception as exc:
                errors.append(exc)

        def querier():
            try:
                qemb = np.random.randn(16).astype(np.float32)
                for _ in range(50):
                    results = field.query(qemb, top_k=5)
                    query_results.append(len(results))
                    time.sleep(0.005)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=querier)
        t3 = threading.Thread(target=querier)
        t1.start()
        t2.start()
        t3.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        t3.join(timeout=30)

        assert not errors, f"Exceptions: {errors}"
        assert len(field.nodes) == 400
        assert all(0 <= r <= 5 for r in query_results)


class TestConcurrentMixed:
    def test_mixed_add_query_delete(self):
        """Interleaved add, query, and delete operations."""
        field = _make_field()
        errors = []

        # Pre-populate
        for i in range(100):
            emb = np.random.randn(16).astype(np.float32)
            field.add_node(embedding=emb, content={"text": f"base{i}"}, node_id=f"base{i}")

        def worker(tid):
            try:
                for i in range(30):
                    emb = np.random.randn(16).astype(np.float32)
                    nid = f"t{tid}_n{i}"
                    field.add_node(embedding=emb, content={"text": nid}, node_id=nid)
                    field.query(emb, top_k=3)
                    if i % 3 == 0:
                        field.delete_nodes([nid])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Exceptions: {errors}"
        # Some nodes deleted, some remain — just verify no crash
        assert len(field.nodes) >= 100  # At least base nodes remain
