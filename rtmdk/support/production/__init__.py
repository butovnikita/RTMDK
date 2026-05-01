"""Production stack for RTMDK."""
from __future__ import annotations

import re
import time
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from rtmdk.nodes import EvalResult


class ShadowModeEvaluator:
    def __init__(self, fallback_threshold: float = 0.3):
        self.fallback_threshold = fallback_threshold
        self._shadow_results: List[Dict] = []
        self._production_results: List[Dict] = []
        self._fallback_count = 0
        self._total_comparisons = 0

    def compare(self, shadow_output: Any, production_output: Any,
                metric_name: str = "response_quality") -> Dict[str, Any]:
        self._shadow_results.append({"value": shadow_output, "metric": metric_name})
        self._production_results.append({"value": production_output, "metric": metric_name})
        self._total_comparisons += 1
        diff = abs(float(shadow_output) - float(production_output))
        is_better = shadow_output > production_output
        if diff > self.fallback_threshold:
            self._fallback_count += 1
        return {
            "shadow_value": shadow_output, "production_value": production_output,
            "difference": diff, "shadow_better": is_better,
            "fallback_triggered": diff > self.fallback_threshold,
        }

    def get_correlation(self) -> float:
        if len(self._shadow_results) < 3:
            return 0.0
        shadow_vals = [r["value"] for r in self._shadow_results]
        prod_vals = [r["value"] for r in self._production_results]
        if np.std(shadow_vals) < 1e-8 or np.std(prod_vals) < 1e-8:
            return 1.0
        corr = np.corrcoef(shadow_vals, prod_vals)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def get_fallback_rate(self) -> float:
        return self._fallback_count / max(self._total_comparisons, 1)


class RAGASPlusEvaluator:
    def __init__(self):
        self._eval_history: List["EvalResult"] = []

    def evaluate(self, question: str, answer: str, contexts: List[str],
                 ground_truth: Optional[str] = None,
                 causal_edges: Optional[List[Tuple[str, str, float]]] = None) -> "EvalResult":
        from rtmdk.nodes import EvalResult
        result = EvalResult()
        result.context_precision = self._compute_context_precision(question, contexts)
        if ground_truth:
            result.context_recall = self._compute_context_recall(ground_truth, contexts)
        else:
            result.context_recall = result.context_precision * 0.8
        result.answer_relevance = self._compute_answer_relevance(question, answer)
        result.faithfulness = self._compute_faithfulness(answer, contexts)
        if causal_edges:
            result.causal_consistency = self._compute_causal_consistency(answer, causal_edges)
        else:
            result.causal_consistency = 0.5
        result.temporal_coherence = self._compute_temporal_coherence(contexts)
        weights = [0.2, 0.15, 0.2, 0.2, 0.15, 0.1]
        scores = [result.context_precision, result.context_recall,
                  result.answer_relevance, result.faithfulness,
                  result.causal_consistency, result.temporal_coherence]
        result.overall_score = sum(w * s for w, s in zip(weights, scores))
        self._eval_history.append(result)
        return result

    def _compute_context_precision(self, question: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
        if not q_tokens:
            return 0.0
        precision_scores = []
        for ctx in contexts:
            c_tokens = set(re.findall(r"\b\w+\b", ctx.lower()))
            if c_tokens:
                precision_scores.append(len(q_tokens & c_tokens) / len(q_tokens))
        return float(np.mean(precision_scores)) if precision_scores else 0.0

    def _compute_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        gt_tokens = set(re.findall(r"\b\w+\b", ground_truth.lower()))
        if not gt_tokens:
            return 0.0
        all_ctx_tokens = set()
        for ctx in contexts:
            all_ctx_tokens.update(re.findall(r"\b\w+\b", ctx.lower()))
        if not all_ctx_tokens:
            return 0.0
        return len(gt_tokens & all_ctx_tokens) / len(gt_tokens)

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
        a_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
        if not q_tokens or not a_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / len(q_tokens)

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        a_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
        if not a_tokens:
            return 0.0
        all_ctx = " ".join(contexts).lower()
        ctx_tokens = set(re.findall(r"\b\w+\b", all_ctx))
        if not ctx_tokens:
            return 0.5
        return len(a_tokens & ctx_tokens) / len(a_tokens)

    def _compute_causal_consistency(self, answer: str,
                                     causal_edges: List[Tuple[str, str, float]]) -> float:
        if not causal_edges:
            return 0.5
        answer_lower = answer.lower()
        consistent = 0
        for cause, effect, strength in causal_edges:
            if cause.lower() in answer_lower and effect.lower() in answer_lower:
                consistent += strength
        return consistent / len(causal_edges) if causal_edges else 0.5

    def _compute_temporal_coherence(self, contexts: List[str]) -> float:
        if len(contexts) < 2:
            return 1.0
        temporal_markers = ["then", "after", "before", "next", "later", "previously"]
        coherent = 0
        for ctx in contexts:
            ctx_lower = ctx.lower()
            if any(m in ctx_lower for m in temporal_markers):
                coherent += 1
        return coherent / len(contexts)

    def get_trend(self) -> Dict[str, float]:
        if len(self._eval_history) < 5:
            return {}
        recent = self._eval_history[-10:]
        older = self._eval_history[-20:-10] if len(self._eval_history) >= 20 else self._eval_history[:5]
        return {
            "recent_overall": np.mean([e.overall_score for e in recent]),
            "older_overall": np.mean([e.overall_score for e in older]),
            "trend": "improving" if np.mean([e.overall_score for e in recent]) > np.mean([e.overall_score for e in older]) else "degrading",
        }


class AutoRollbackManager:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self._baseline_score: Optional[float] = None
        self._recent_scores: deque = deque(maxlen=50)
        self._rollback_count = 0
        self._last_rollback_time: float = 0
        self._cooldown_period: float = 300.0

    def set_baseline(self, score: float):
        self._baseline_score = score

    def record_score(self, score: float) -> bool:
        self._recent_scores.append(score)
        if self._baseline_score is None or len(self._recent_scores) < 10:
            return False
        if time.time() - self._last_rollback_time < self._cooldown_period:
            return False
        recent_mean = np.mean(self._recent_scores)
        degradation = self._baseline_score - recent_mean
        if degradation > self.threshold:
            self._rollback_count += 1
            self._last_rollback_time = time.time()
            return True
        return False

    def get_rollback_rate(self) -> float:
        return self._rollback_count / max(len(self._recent_scores), 1)

    def get_state(self) -> Dict:
        return {
            "baseline_score": self._baseline_score,
            "recent_mean": float(np.mean(self._recent_scores)) if self._recent_scores else 0,
            "rollback_count": self._rollback_count,
            "rollback_rate": self.get_rollback_rate(),
        }
