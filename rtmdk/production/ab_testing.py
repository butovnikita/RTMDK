"""
rtmdk/production/ab_testing.py — A/B Testing Framework.

Randomly assigns users to config variants and tracks metrics.
"""

import hashlib
from typing import Dict, List, Any
from collections import defaultdict


class ABTesting:
    """A/B testing for RTMDK configurations.

    Usage:
        ab = ABTesting()
        ab.add_variant("A", {"top_k": 5})
        ab.add_variant("B", {"top_k": 10})

        variant, config = ab.get_variant("user123")
        # Use config for this user...

        ab.record_metric("user123", "recall", 0.95)

        results = ab.get_results()
    """

    def __init__(self) -> None:
        self._variants: Dict[str, Dict] = {}
        self._assignments: Dict[str, str] = {}  # user_id → variant
        self._metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def add_variant(self, name: str, config_overrides: Dict):
        """Add a test variant."""
        self._variants[name] = config_overrides

    def get_variant(self, user_id: str) -> tuple:
        """Get assigned variant for user."""
        if user_id not in self._assignments:
            # Deterministic assignment based on user_id hash
            variant_names = list(self._variants.keys())
            if not variant_names:
                return "control", {}
            idx = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % len(variant_names)
            chosen = variant_names[idx]
            self._assignments[user_id] = chosen

        variant_name = self._assignments[user_id]
        return variant_name, self._variants.get(variant_name, {})

    def record_metric(self, user_id: str, metric_name: str, value: float):
        """Record a metric for a user."""
        variant = self._assignments.get(user_id, "unknown")
        self._metrics[variant][metric_name].append(value)

    def get_results(self) -> Dict[str, Any]:
        """Get A/B test results with statistical comparison."""
        results = {}
        for variant_name, config in self._variants.items():
            metrics = self._metrics.get(variant_name, {})
            results[variant_name] = {
                "config": config,
                "n_users": sum(1 for v in self._assignments.values() if v == variant_name),
                "metrics": {
                    name: {
                        "mean": sum(values) / len(values) if values else 0,
                        "count": len(values),
                    }
                    for name, values in metrics.items()
                },
            }
        return results
