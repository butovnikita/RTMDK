"""Tests for rtmdk.production.llm_eval."""

from unittest.mock import patch, MagicMock
import numpy as np
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
from rtmdk.production.llm_eval import LLMEvaluator


def _embed(text: str) -> np.ndarray:
    return np.random.randn(768).astype(np.float32)


def _make_mem():
    cfg = RTMDKConfig(latent_dim=64)
    return RTMDKMemory(config=cfg, embedder=_embed)


class TestLLMEvaluator:
    def test_no_api_key(self):
        mem = _make_mem()
        evaluator = LLMEvaluator(memory=mem, api_key="")
        result = evaluator.evaluate_query("What is X?", "X is Y")
        assert result["llm_answer"] == "[No API key]"
        assert "exact_match" in result
        assert "hallucination_rate" in result

    @patch("requests.post")
    def test_with_api_key(self, mock_post):
        mock_post.return_value = MagicMock(
            json=lambda: {
                "choices": [{"message": {"content": "The answer is Y"}}]
            }
        )
        mem = _make_mem()
        evaluator = LLMEvaluator(memory=mem, api_key="test-key")
        result = evaluator.evaluate_query("What is X?", "The answer is Y")
        assert result["llm_answer"] == "The answer is Y"
        assert result["exact_match"] == 1.0
        assert result["context_match"] is False

    def test_evaluate_dataset_empty(self):
        mem = _make_mem()
        evaluator = LLMEvaluator(memory=mem, api_key="")
        agg = evaluator.evaluate_dataset([])
        assert agg == {"count": 0}

    @patch("requests.post")
    def test_evaluate_dataset(self, mock_post, tmp_path):
        mock_post.return_value = MagicMock(
            json=lambda: {
                "choices": [{"message": {"content": "answer"}}]
            }
        )
        mem = _make_mem()
        evaluator = LLMEvaluator(memory=mem, api_key="test-key")
        dataset = [
            {"query": "q1", "answer": "answer"},
            {"query": "q2", "answer": "other"},
        ]
        ckpt = str(tmp_path / "ckpt.json")
        agg = evaluator.evaluate_dataset(dataset, checkpoint_every=1, checkpoint_path=ckpt)
        assert agg["count"] == 2
        assert "avg_exact_match" in agg

    def test_aggregate_results_empty(self):
        mem = _make_mem()
        evaluator = LLMEvaluator(memory=mem, api_key="")
        assert evaluator._aggregate_results() == {"count": 0}

    def test_get_results(self):
        mem = _make_mem()
        evaluator = LLMEvaluator(memory=mem, api_key="")
        evaluator.evaluate_query("q", "a")
        assert len(evaluator.get_results()) == 1
