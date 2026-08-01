"""Tests for rtmdk/memory/adaptive_pc.py — phase coupling estimation."""

import numpy as np

from rtmdk.memory.adaptive_pc import estimate_optimal_pc

DIM = 8


def unit(v):
    return v / np.linalg.norm(v)


def doc_with_sim(query, sim, rng):
    """Build a normalized doc vector with given cosine similarity to query."""
    perp = rng.standard_normal(DIM)
    perp -= perp @ query * query
    perp /= np.linalg.norm(perp)
    return unit(sim * query + np.sqrt(max(1.0 - sim**2, 0.0)) * perp)


class TestEmptyInputs:
    def test_no_queries(self):
        assert estimate_optimal_pc(np.ones((2, DIM)), np.zeros((0, DIM))) == 0.0

    def test_no_docs(self):
        assert estimate_optimal_pc(np.zeros((0, DIM)), np.ones((2, DIM))) == 0.0


class TestWithGroundTruth:
    def test_easy_dataset_returns_zero(self):
        rng = np.random.default_rng(0)
        queries = np.array([unit(rng.standard_normal(DIM)) for _ in range(6)])
        docs = queries.copy()  # top-1 always correct
        targets = np.arange(6)

        assert estimate_optimal_pc(docs, queries, targets) == 0.0

    def test_moderately_hard_returns_01(self):
        rng = np.random.default_rng(1)
        queries = np.array([unit(rng.standard_normal(DIM)) for _ in range(4)])
        docs = np.vstack([queries, unit(rng.standard_normal(DIM))])
        # 2 of 4 targets correct → top1_acc = 0.5
        targets = np.array([0, 1, 4, 4])

        assert estimate_optimal_pc(docs, queries, targets) == 0.1

    def test_very_hard_returns_015(self):
        rng = np.random.default_rng(2)
        queries = np.array([unit(rng.standard_normal(DIM)) for _ in range(4)])
        docs = np.vstack([queries, unit(rng.standard_normal(DIM))])
        # 1 of 4 correct → top1_acc = 0.25
        targets = np.array([0, 4, 4, 4])

        assert estimate_optimal_pc(docs, queries, targets) == 0.15


class TestGapHeuristic:
    def test_large_gap_returns_zero(self):
        rng = np.random.default_rng(3)
        q = unit(rng.standard_normal(DIM))
        docs = np.array([q.copy(), doc_with_sim(q, 0.5, rng)])  # gap = 1 - 0.5 = 0.5

        assert estimate_optimal_pc(docs, q[np.newaxis, :]) == 0.0

    def test_medium_gap_returns_01(self):
        rng = np.random.default_rng(4)
        q = unit(rng.standard_normal(DIM))
        docs = np.array([doc_with_sim(q, 0.7, rng), doc_with_sim(q, 0.5, rng)])  # gap ≈ 0.2

        assert estimate_optimal_pc(docs, q[np.newaxis, :]) == 0.1

    def test_small_gap_returns_015(self):
        rng = np.random.default_rng(5)
        q = unit(rng.standard_normal(DIM))
        docs = np.array([doc_with_sim(q, 0.6, rng), doc_with_sim(q, 0.55, rng)])  # gap ≈ 0.05

        assert estimate_optimal_pc(docs, q[np.newaxis, :]) == 0.15

    def test_result_within_documented_range(self):
        rng = np.random.default_rng(6)
        docs = np.array([unit(rng.standard_normal(DIM)) for _ in range(20)])
        queries = np.array([unit(rng.standard_normal(DIM)) for _ in range(10)])

        pc = estimate_optimal_pc(docs, queries)
        assert 0.0 <= pc <= 0.3
