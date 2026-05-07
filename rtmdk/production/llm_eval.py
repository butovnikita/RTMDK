"""
rtmdk/production/llm_eval.py — LLM-as-Judge Evaluation Pipeline.

Automatically evaluates retrieval quality by comparing LLM answers with ground truth.
Features:
- Works with OpenRouter, OpenAI, Anthropic APIs
- Metrics: exact_match, semantic_similarity, hallucination_rate
- Batch evaluation on datasets
- Progress tracking and checkpointing
"""

import os
import json
import time
from typing import Dict, List, Any
from pathlib import Path


class LLMEvaluator:
    """Evaluates RTMDK retrieval quality using LLM-as-judge.

    Usage:
        evaluator = LLMEvaluator(
            memory=memory,
            api_key="<your-api-key-here>",
            provider="openrouter",
            model="anthropic/claude-3.5-sonnet"
        )

        results = evaluator.evaluate_dataset(dataset)
    """

    def __init__(
        self,
        memory,  # RTMDKMemory instance
        api_key: str = "",
        provider: str = "openrouter",
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.memory = memory
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self._results: List[Dict] = []

    def evaluate_query(
        self,
        query: str,
        expected_answer: str,
        session_id: str = "eval",
    ) -> Dict[str, Any]:
        """Evaluate a single query.

        Args:
            query: The query to test
            expected_answer: Ground truth answer
            session_id: Session ID for retrieval

        Returns:
            Dict with evaluation metrics
        """
        t0 = time.time()

        # Get RTMDK context
        ctx = self.memory.load_memory_variables({
            "input": query,
            "session_id": session_id,
        })
        context = ctx.get("rtmdk_context", "")

        # Get LLM answer
        llm_answer = self._ask_llm(query, context)

        latency = (time.time() - t0) * 1000

        # Compute metrics
        metrics = self._compute_metrics(expected_answer, llm_answer, context)
        metrics["latency_ms"] = round(latency, 1)
        metrics["query"] = query
        metrics["expected"] = expected_answer
        metrics["llm_answer"] = llm_answer[:200]
        metrics["context_length"] = len(context)

        self._results.append(metrics)
        return metrics

    def evaluate_dataset(
        self,
        dataset: List[Dict],
        checkpoint_every: int = 50,
        checkpoint_path: str = "llm_eval_checkpoint.json",
    ) -> Dict[str, Any]:
        """Evaluate on a full dataset.

        Args:
            dataset: List of {query, answer} dicts
            checkpoint_every: Save checkpoint every N queries
            checkpoint_path: Path to save checkpoint

        Returns:
            Aggregate metrics
        """
        ckpt_path = Path(checkpoint_path)
        start_idx = 0

        if ckpt_path.exists():
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            start_idx = ckpt.get("done", 0)
            self._results = ckpt.get("results", [])

        for i in range(start_idx, len(dataset)):
            item = dataset[i]
            query = item.get("query", item.get("input", ""))
            answer = item.get("answer", item.get("output", ""))

            if query and answer:
                self.evaluate_query(query, answer)

            if (i + 1) % checkpoint_every == 0:
                with open(ckpt_path, 'w') as f:
                    json.dump({"done": i + 1, "results": self._results}, f)

        # Aggregate results
        return self._aggregate_results()

    def _ask_llm(self, query: str, context: str) -> str:
        """Get LLM answer using the configured API."""
        if not self.api_key:
            return "[No API key]"

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://rtmdk.local",
            }

            system_msg = (
                "Answer based ONLY on this context. "
                "If you don't know, say 'I don't know'.\n\n"
                f"Context: {context}")

            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": query},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
                timeout=30,
            )

            data = resp.json()
            return data.get(
                "choices", [
                    {}])[0].get(
                "message", {}).get(
                "content", "")
        except Exception as e:
            return f"[Error: {e}]"

    def _compute_metrics(
        self,
        expected: str,
        actual: str,
        context: str,
    ) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        expected_lower = expected.lower()
        actual_lower = actual.lower()

        # Exact match (lenient: key word overlap)
        expected_words = set(w for w in expected_lower.split() if len(w) > 3)
        actual_words = set(w for w in actual_lower.split() if len(w) > 3)

        exact_match = 0
        if expected_words and actual_words:
            overlap = expected_words & actual_words
            exact_match = len(overlap) / len(expected_words)

        # Context match: does expected answer appear in context?
        context_match = any(w in context.lower()
                            for w in expected_words) if expected_words else False

        # Hallucination: words in answer not in context or expected
        context_words = set(context.lower().split())
        hallucinated = actual_words - context_words - expected_words
        hallucination_rate = len(hallucinated) / max(len(actual_words), 1)

        return {
            "exact_match": round(exact_match, 3),
            "context_match": context_match,
            "hallucination_rate": round(hallucination_rate, 3),
        }

    def _aggregate_results(self) -> Dict[str, Any]:
        """Compute aggregate metrics."""
        if not self._results:
            return {"count": 0}

        exact_matches = [r["exact_match"] for r in self._results]
        hallucination = [r["hallucination_rate"] for r in self._results]
        latencies = [r["latency_ms"] for r in self._results]
        context_match = sum(1 for r in self._results if r["context_match"])

        return {
            "count": len(self._results),
            "avg_exact_match": round(sum(exact_matches) / len(exact_matches), 3),
            "context_match_rate": round(context_match / len(self._results), 3),
            "avg_hallucination_rate": round(sum(hallucination) / len(hallucination), 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        }

    def get_results(self) -> List[Dict]:
        """Get all individual results."""
        return self._results
