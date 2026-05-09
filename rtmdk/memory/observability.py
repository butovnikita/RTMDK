"""Observability: latency tracking, metrics, alerting, and export for RTMDK.

Lightweight implementation without external deps (no OpenTelemetry agent
required).  Metrics are stored in-memory and can be exported periodically.
"""
from __future__ import annotations
import json
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


class AlertHandler:
    """Base class for alert handlers."""

    def __call__(self, alert_name: str, value: float) -> None:
        raise NotImplementedError


class WebhookAlertHandler(AlertHandler):
    """Send alerts to a webhook URL."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self.url = url
        self.headers = headers or {}

    def __call__(self, alert_name: str, value: float) -> None:
        try:
            import urllib.request
            payload = json.dumps({
                "alert": alert_name,
                "value": value,
                "timestamp": time.time(),
            }).encode("utf-8")
            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={**self.headers, "Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            logger.warning("Webhook alert failed for %s", alert_name, exc_info=True)


class SlackAlertHandler(AlertHandler):
    """Send alerts to Slack via incoming webhook."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def __call__(self, alert_name: str, value: float) -> None:
        try:
            import urllib.request
            payload = json.dumps({
                "text": f"🚨 RTMDK Alert: *{alert_name}* | value={value:.2f}",
            }).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            logger.warning("Slack alert failed for %s", alert_name, exc_info=True)


class PagerDutyAlertHandler(AlertHandler):
    """Send alerts to PagerDuty Events API v2."""

    def __init__(self, routing_key: str, severity: str = "warning"):
        self.routing_key = routing_key
        self.severity = severity

    def __call__(self, alert_name: str, value: float) -> None:
        try:
            import urllib.request
            payload = json.dumps({
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"RTMDK alert: {alert_name}={value:.2f}",
                    "severity": self.severity,
                    "source": "rtmdk",
                    "custom_details": {"value": value},
                },
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://events.pagerduty.com/v2/enqueue",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            logger.warning("PagerDuty alert failed for %s", alert_name, exc_info=True)


class MemoryMetrics:
    """In-memory metrics collector for RTMDK retrieval pipeline."""

    def __init__(self):
        self.query_latency = LatencyTracker()
        self.ingestion_latency = LatencyTracker()
        self.query_cache_hits = 0
        self.query_cache_misses = 0
        self.consolidation_count = 0
        self.alert_rules: List[AlertRule] = []
        self._alert_handlers: List[AlertHandler] = []
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

    def add_alert_handler(self, handler: AlertHandler) -> None:
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

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        p = self.query_latency.percentiles()
        lines.append(f'rtmdk_query_latency_p50 {p["p50"]:.3f}')
        lines.append(f'rtmdk_query_latency_p95 {p["p95"]:.3f}')
        lines.append(f'rtmdk_query_latency_p99 {p["p99"]:.3f}')
        lines.append(f'rtmdk_cache_hit_ratio {self.cache_hit_ratio():.3f}')
        lines.append(f'rtmdk_consolidation_count {self.consolidation_count}')
        return "\n".join(lines)

    def flush_to_file(self, path: str) -> None:
        """Write snapshot to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, indent=2)
