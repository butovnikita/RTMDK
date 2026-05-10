"""A/B testing framework for pipeline vs legacy retrieval.

Compares retrieve_nodes_pipeline() against legacy retrieve_nodes()
on latency, result overlap, and ranking correlation.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rtmdk.pipeline.base import PipelineContext


class PipelineABTester:
    """A/B test pipeline retrieval against legacy path.

    Usage:
        tester = PipelineABTester(memory)
        result = tester.compare_single("query", top_k=5)
        print(result["latency_ms"]["legacy"], result["latency_ms"]["pipeline"])

        batch = tester.compare_batch(["q1", "q2", "q3"], top_k=5)
        summary = tester.summary()
    """

    def __init__(self, memory: Any):
        self.memory = memory
        self._runs: List[Dict[str, Any]] = []

    def compare_single(
        self,
        query: str,
        top_k: int = 5,
        session_id: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Run both legacy and pipeline retrieval for a single query.

        Returns:
            Dict with legacy results, pipeline results, latency, and comparison metrics.
        """
        if embedding is None:
            embedding = self.memory.embedder(query)

        # Legacy path
        t0 = time.perf_counter()
        legacy_results = self.memory.retrieve_nodes(
            query, embedding, top_k=top_k, session_id=session_id
        )
        legacy_latency_ms = (time.perf_counter() - t0) * 1000

        # Pipeline path
        t0 = time.perf_counter()
        pipeline_output = self.memory.retrieve_nodes_pipeline(
            query, embedding=embedding, top_k=top_k, session_id=session_id
        )
        pipeline_latency_ms = (time.perf_counter() - t0) * 1000
        pipeline_results = pipeline_output["results"]

        result = {
            "query": query,
            "top_k": top_k,
            "legacy": {
                "results": legacy_results,
                "latency_ms": round(legacy_latency_ms, 3),
                "result_ids": [r[0] for r in legacy_results],
            },
            "pipeline": {
                "results": pipeline_results,
                "latency_ms": round(pipeline_latency_ms, 3),
                "result_ids": [r[0] for r in pipeline_results],
                "route": pipeline_output.get("route"),
                "metrics": pipeline_output.get("metrics"),
            },
            "comparison": self._compare(
                legacy_results, pipeline_results, legacy_latency_ms, pipeline_latency_ms
            ),
        }
        self._runs.append(result)
        return result

    def compare_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run A/B comparison for multiple queries."""
        return [
            self.compare_single(q, top_k=top_k, session_id=session_id)
            for q in queries
        ]

    def _compare(
        self,
        legacy: List[Tuple[str, float, Any]],
        pipeline: List[Tuple[str, float, Any]],
        legacy_latency_ms: float,
        pipeline_latency_ms: float,
    ) -> Dict[str, Any]:
        """Compute comparison metrics between legacy and pipeline results."""
        legacy_ids = [r[0] for r in legacy]
        pipeline_ids = [r[0] for r in pipeline]

        # Jaccard overlap
        set_legacy = set(legacy_ids)
        set_pipeline = set(pipeline_ids)
        intersection = set_legacy & set_pipeline
        union = set_legacy | set_pipeline
        jaccard = len(intersection) / max(len(union), 1)

        # Ranking correlation (Kendall tau simplified)
        # Only compare overlapping items
        overlap_ids = list(intersection)
        if len(overlap_ids) >= 2:
            legacy_ranks = {nid: i for i, nid in enumerate(legacy_ids) if nid in intersection}
            pipeline_ranks = {nid: i for i, nid in enumerate(pipeline_ids) if nid in intersection}
            legacy_order = [legacy_ranks[nid] for nid in overlap_ids]
            pipeline_order = [pipeline_ranks[nid] for nid in overlap_ids]
            # Simplified Kendall tau: count concordant / total pairs
            n = len(overlap_ids)
            concordant = 0
            total_pairs = n * (n - 1) // 2
            for i in range(n):
                for j in range(i + 1, n):
                    if (legacy_order[i] - legacy_order[j]) * (pipeline_order[i] - pipeline_order[j]) > 0:
                        concordant += 1
            kendall_tau = concordant / max(total_pairs, 1) if total_pairs > 0 else 1.0
        else:
            kendall_tau = 1.0 if len(overlap_ids) <= 1 else 0.0

        # Latency delta
        latency_delta_ms = pipeline_latency_ms - legacy_latency_ms
        latency_ratio = pipeline_latency_ms / max(legacy_latency_ms, 0.001)

        return {
            "jaccard_overlap": round(jaccard, 4),
            "kendall_tau": round(kendall_tau, 4),
            "overlap_count": len(intersection),
            "legacy_only_count": len(set_legacy - set_pipeline),
            "pipeline_only_count": len(set_pipeline - set_legacy),
            "latency_delta_ms": round(latency_delta_ms, 3),
            "latency_ratio": round(latency_ratio, 3),
            "faster": "pipeline" if pipeline_latency_ms < legacy_latency_ms else "legacy",
        }

    def summary(self) -> Dict[str, Any]:
        """Aggregate statistics across all runs."""
        if not self._runs:
            return {"runs": 0}

        legacy_latencies = [r["legacy"]["latency_ms"] for r in self._runs]
        pipeline_latencies = [r["pipeline"]["latency_ms"] for r in self._runs]
        jaccards = [r["comparison"]["jaccard_overlap"] for r in self._runs]
        kendalls = [r["comparison"]["kendall_tau"] for r in self._runs]

        def _stats(values: List[float]) -> Dict[str, float]:
            arr = np.array(values)
            return {
                "mean": round(float(np.mean(arr)), 3),
                "median": round(float(np.median(arr)), 3),
                "p95": round(float(np.percentile(arr, 95)), 3),
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
            }

        pipeline_faster = sum(
            1 for r in self._runs
            if r["pipeline"]["latency_ms"] < r["legacy"]["latency_ms"]
        )

        return {
            "runs": len(self._runs),
            "legacy_latency_ms": _stats(legacy_latencies),
            "pipeline_latency_ms": _stats(pipeline_latencies),
            "jaccard_overlap": _stats(jaccards),
            "kendall_tau": _stats(kendalls),
            "pipeline_faster_count": pipeline_faster,
            "legacy_faster_count": len(self._runs) - pipeline_faster,
        }
