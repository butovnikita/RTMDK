"""Pipeline cost analyzer — estimates per-query compute cost.

Tracks:
- Embedding API calls (or local GPU time)
- Reranker inference cost
- Explanation generation cost
- Total latency budget consumption

Useful for:
- Capacity planning
- Rate-limiting by cost, not just RPS
- Showback / chargeback in multi-tenant deployments
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Per-query cost breakdown."""

    query_text: str
    stage_costs: Dict[str, float] = field(default_factory=dict)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return {
            "query_text": self.query_text[:50],
            "stage_costs": {k: round(v, 6) for k, v in self.stage_costs.items()},
            "total_cost": round(self.total_cost, 6),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


class PipelineCostAnalyzer:
    """Analyzes and tracks pipeline query costs.

    Usage:
        analyzer = PipelineCostAnalyzer()
        analyzer.record_stage("embed", latency_ms=15.0, tokens=12)
        analyzer.record_stage("rerank", latency_ms=8.0, tokens=0)
        cost = analyzer.finalize("What is AI?")
        print(cost.total_cost)
    """

    # Default cost model: cost units per stage
    # 1 unit ≈ $0.0001 (roughly 1 SBERT inference on CPU)
    DEFAULT_COST_RATES: Dict[str, float] = {
        "embed": 0.50,  # GPU/CPU intensive
        "route": 0.01,  # Negligible
        "retrieve": 0.02,  # Vector search
        "rerank": 0.30,  # Cross-encoder or sentence model
        "calibrate": 0.01,  # Simple arithmetic
        "explain": 0.15,  # Template or small LLM call
    }

    def __init__(self, cost_rates: Optional[Dict[str, float]] = None):
        self.cost_rates = cost_rates or dict(self.DEFAULT_COST_RATES)
        self._history: List[CostBreakdown] = []
        self._current: Optional[CostBreakdown] = None

    def start(self, query_text: str) -> None:
        """Begin tracking a new query."""
        self._current = CostBreakdown(query_text=query_text)

    def record_stage(
        self,
        stage_name: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Record cost for a completed stage."""
        if self._current is None:
            raise RuntimeError("Call start() before record_stage()")
        rate = self.cost_rates.get(stage_name, 0.0)
        # Latency-adjusted cost: slower stages cost more (opportunity cost)
        cost = rate * (1 + latency_ms / 1000.0)
        self._current.stage_costs[stage_name] = cost
        self._current.total_cost += cost
        self._current.total_latency_ms += latency_ms
        self._current.tokens_in += tokens_in
        self._current.tokens_out += tokens_out

    def finalize(self, query_text: Optional[str] = None) -> CostBreakdown:
        """Finalize tracking and return the cost breakdown."""
        if self._current is None:
            raise RuntimeError("Call start() before finalize()")
        if query_text:
            self._current.query_text = query_text
        result = self._current
        self._history.append(result)
        self._current = None
        return result

    def summary(self, n: Optional[int] = None) -> Dict[str, float]:
        """Aggregate cost summary over tracked history."""
        history = self._history[-n:] if n else self._history
        if not history:
            return {}
        total_cost = sum(h.total_cost for h in history)
        total_latency = sum(h.total_latency_ms for h in history)
        result: Dict[str, Any] = {
            "queries": len(history),
            "total_cost": round(total_cost, 6),
            "avg_cost_per_query": round(total_cost / len(history), 6),
            "total_latency_ms": round(total_latency, 3),
            "avg_latency_ms": round(total_latency / len(history), 3),
            "cost_by_stage": {
                stage: round(sum(h.stage_costs.get(stage, 0.0) for h in history), 6) for stage in self.cost_rates.keys()
            },
        }
        return result

    def reset(self) -> None:
        """Clear history."""
        self._history.clear()
        self._current = None
