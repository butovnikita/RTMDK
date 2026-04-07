"""
test_rtmdk_eval.py
Tests for eval_pipeline and streamlit_app components.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_pipeline import (
    generate_continual_qa_dataset,
    generate_long_bench_dataset,
    generate_memory_bench_dataset,
    EvalMetrics,
    ContinualEvalPipeline,
)

from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory


@pytest.fixture
def dummy_embedder():
    def _embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        base = np.random.randn(768).astype(np.float32) * 0.1
        sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
        base[:10] = sig
        return base
    return _embed


@pytest.fixture
def eval_memory(dummy_embedder):
    config = RTMDKConfig(
        embedding_dim=768, latent_dim=64, top_k=5, enable_async=False,
        min_response=0.01,
    )
    return RTMDKMemory(config=config, embedder=dummy_embedder)


# ============================================================================
# DATASET GENERATION
# ============================================================================

class TestDatasetGeneration:
    def test_continual_qa_dataset(self):
        data = generate_continual_qa_dataset(n_samples=10)
        assert len(data) == 10
        assert all("topic" in d for d in data)
        assert all("question" in d for d in data)
        assert all("answer" in d for d in data)

    def test_long_bench_dataset(self):
        data = generate_long_bench_dataset(n_samples=5)
        assert len(data) == 5
        assert all("context" in d for d in data)
        assert all("context_length" in d for d in data)
        assert all(d["context_length"] >= 500 for d in data)

    def test_memory_bench_dataset(self):
        data = generate_memory_bench_dataset(n_samples=8)
        assert len(data) == 8
        assert all("entity" in d for d in data)
        assert all("attribute" in d for d in data)
        assert all("fact" in d for d in data)


# ============================================================================
# EVAL METRICS
# ============================================================================

class TestEvalMetrics:
    def test_creation(self):
        m = EvalMetrics()
        assert len(m.results) == 0

    def test_record(self):
        m = EvalMetrics()
        result = m.record("test_bench", "sample_1", 0.8, 10.0, True)
        assert result["accuracy"] == 0.8
        assert result["latency_ms"] == 10.0
        assert result["context_used"] is True
        assert len(m.results) == 1

    def test_compute_forgetting(self):
        m = EvalMetrics()
        initial = {"s1": 0.9, "s2": 0.8}
        m.record("bench", "s1", 0.7, 5.0)
        m.record("bench", "s2", 0.6, 5.0)
        forgetting = m.compute_forgetting("bench", initial)
        assert abs(forgetting["s1"] - 0.2) < 1e-10
        assert abs(forgetting["s2"] - 0.2) < 1e-10

    def test_compute_interference(self):
        m = EvalMetrics()
        m.record("bench", "s1", 0.5, 5.0)
        m.record("bench", "s2", 0.7, 5.0)
        interference = m.compute_interference("bench", "topic_a")
        assert 0.0 <= interference <= 1.0

    def test_get_summary(self):
        m = EvalMetrics()
        m.record("bench", "s1", 0.8, 10.0, True)
        m.record("bench", "s2", 0.6, 15.0, False)
        summary = m.get_summary()
        assert summary["n_evaluations"] == 2
        assert summary["overall_accuracy"] == 0.7
        assert "bench_accuracy" in summary

    def test_get_summary_empty(self):
        m = EvalMetrics()
        summary = m.get_summary()
        assert summary["n_evaluations"] == 0


# ============================================================================
# EVAL PIPELINE
# ============================================================================

class TestContinualEvalPipeline:
    def test_creation(self, eval_memory):
        pipeline = ContinualEvalPipeline(eval_memory)
        assert pipeline.memory is eval_memory
        assert pipeline.metrics is not None

    def test_run_continual_qa(self, eval_memory):
        pipeline = ContinualEvalPipeline(eval_memory)
        data = generate_continual_qa_dataset(n_samples=5)
        result = pipeline.run_continual_qa(data)
        assert "n_samples" in result
        assert "accuracy" in result
        assert result["n_samples"] == 5

    def test_run_long_bench(self, eval_memory):
        pipeline = ContinualEvalPipeline(eval_memory)
        data = generate_long_bench_dataset(n_samples=3)
        result = pipeline.run_long_bench(data)
        assert result["n_samples"] == 3

    def test_run_memory_bench(self, eval_memory):
        pipeline = ContinualEvalPipeline(eval_memory)
        data = generate_memory_bench_dataset(n_samples=5)
        result = pipeline.run_memory_bench(data)
        assert result["n_samples"] == 5

    def test_compute_forgetting(self, eval_memory):
        pipeline = ContinualEvalPipeline(eval_memory)
        data = generate_continual_qa_dataset(n_samples=5)
        pipeline.run_continual_qa(data)
        forgetting = pipeline.compute_forgetting()
        assert "continual_qa" in forgetting

    def test_check_rollback_trigger(self, eval_memory):
        pipeline = ContinualEvalPipeline(eval_memory)
        # No data yet, should not trigger
        assert pipeline.check_rollback_trigger() is False

    def test_get_full_report(self, eval_memory):
        pipeline = ContinualEvalPipeline(eval_memory)
        data = generate_continual_qa_dataset(n_samples=3)
        pipeline.run_continual_qa(data)
        report = pipeline.get_full_report()
        assert "summary" in report
        assert "forgetting" in report
        assert "rollback_triggered" in report
        assert "memory_stats" in report
        assert "nodes" in report["memory_stats"]
