"""Observability: latency tracking, metrics, and alerting for RTMDK.

Lightweight implementation without external deps (no OpenTelemetry agent
required).  Metrics are stored in-memory and can be exported periodically.
"""
from __future__ import annotations
import time
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Threshold-based alert rule."""
    name: str
    metric: str
    threshold: float
    comparison: str = "gt"  # "gt" | "lt"
    cooldown: float = 60.0  # seconds between repeated alerts
    last_fired: float = field(default=0.0)

    def check(self, value: float, now: float) -> bool:
        triggered = (self.comparison == "gt" and value > self.threshold) or \
                    (self.comparison == "lt" and value < self.threshold)
        if triggered and (now - self.last_fired) > self.cooldown:
            self.last_fired = now
            return True
        return False


class LatencyTracker:
    """Sliding-window latency histogram with percentile computation."""

    def __init__(self, window_size: int = 1000):
        self._window: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._window.append(latency_ms)

    def percentiles(self) -> Dict[str, float]:
        with self._lock:
            if not self._window:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            arr = np.array(self._window)
            return {
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }

    def mean(self) -> float:
        with self._lock:
            if not self._window:
                return 0.0
            return float(np.mean(self._window))


class MemoryMetrics:
    """In-memory metrics collector for RTMDK retrieval pipeline."""

    def __init__(self):
        self.query_latency = LatencyTracker()
        self.ingestion_latency = LatencyTracker()
        self.query_cache_hits = 0
        self.query_cache_misses = 0
        self.consolidation_count = 0
        self.alert_rules: List[AlertRule] = []
        self._alert_handlers: List[Callable[[str, float], None]] = []
        self._lock = threading.Lock()

    def record_query(self, latency_ms: float, cache_hit: bool) -> None:
        self.query_latency.record(latency_ms)
        with self._lock:
            if cache_hit:
                self.query_cache_hits += 1
            else:
                self.query_cache_misses += 1

    def record_ingestion(self, latency_ms: float) -> None:
        self.ingestion_latency.record(latency_ms)

    def record_consolidation(self) -> None:
        with self._lock:
            self.consolidation_count += 1

    def cache_hit_ratio(self) -> float:
        with self._lock:
            total = self.query_cache_hits + self.query_cache_misses
            if total == 0:
                return 0.0
            return self.query_cache_hits / total

    def add_alert_rule(self, rule: AlertRule) -> None:
        with self._lock:
            self.alert_rules.append(rule)

    def add_alert_handler(self, handler: Callable[[str, float], None]) -> None:
        with self._lock:
            self._alert_handlers.append(handler)

    def check_alerts(self) -> List[str]:
        """Check all alert rules and return triggered messages."""
        triggered = []
        now = time.time()
        metrics = {
            "query_p99": self.query_latency.percentiles()["p99"],
            "query_mean": self.query_latency.mean(),
            "cache_hit_ratio": self.cache_hit_ratio(),
            "consolidation_count": self.consolidation_count,
        }
        with self._lock:
            for rule in self.alert_rules:
                value = metrics.get(rule.metric, 0.0)
                if rule.check(value, now):
                    msg = f"ALERT [{rule.name}]: {rule.metric}={value:.2f} (threshold={rule.threshold})"
                    triggered.append(msg)
                    for handler in self._alert_handlers:
                        try:
                            handler(rule.name, value)
                        except Exception:
                            pass
        return triggered

    def snapshot(self) -> Dict:
        """Return full metrics snapshot."""
        return {
            "query_latency_ms": self.query_latency.percentiles(),
            "query_mean_ms": self.query_latency.mean(),
            "ingestion_latency_ms": self.ingestion_latency.percentiles(),
            "cache_hit_ratio": self.cache_hit_ratio(),
            "consolidation_count": self.consolidation_count,
        }
