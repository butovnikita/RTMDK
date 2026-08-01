"""Health monitoring and SLO enforcement for pipeline stages."""

from __future__ import annotations
from typing import Any, Dict, List, Optional

from rtmdk.pipeline.circuit_breaker import CircuitBreaker


class PipelineHealthMonitor:
    """Tracks per-stage health and manages circuit breakers.

    Usage:
        monitor = PipelineHealthMonitor()
        monitor.set_threshold("rerank", latency_ms=200.0)
        breaker = monitor.get_breaker("rerank")
        stage.circuit_breaker = breaker
    """

    def __init__(self):
        self.thresholds: Dict[str, float] = {}
        self._breakers: Dict[str, CircuitBreaker] = {}

    def set_threshold(self, stage_name: str, latency_ms: float) -> None:
        """Set SLO latency threshold for a stage."""
        self.thresholds[stage_name] = latency_ms

    def get_breaker(
        self,
        stage_name: str,
        failure_threshold: int = 5,
        latency_violation_threshold: int = 3,
        recovery_timeout_ms: float = 30_000.0,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a stage."""
        if stage_name not in self._breakers:
            self._breakers[stage_name] = CircuitBreaker(
                name=stage_name,
                failure_threshold=failure_threshold,
                latency_threshold_ms=self.thresholds.get(stage_name, 500.0),
                latency_violation_threshold=latency_violation_threshold,
                recovery_timeout_ms=recovery_timeout_ms,
            )
        return self._breakers[stage_name]

    def check_stage(self, stage_name: str, latency_ms: float, error: Optional[str]) -> str:
        """Return health status: 'healthy', 'degraded', or 'failed'."""
        if error:
            return "failed"
        if latency_ms > self.thresholds.get(stage_name, 100.0):
            return "degraded"
        return "healthy"

    def check_alerts(
        self,
        ctx,
        degraded_threshold: int = 2,
        latency_threshold_ms: float = 5000.0,
        error_rate_threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Check pipeline context against alert thresholds.

        Returns list of alert dicts with severity and message.
        """
        alerts: List[Dict[str, Any]] = []
        metrics = getattr(ctx, "metrics", [])
        total_latency = sum(m.latency_ms for m in metrics)

        # Check degraded stages count
        degraded = [m for m in metrics if m.degraded]
        if len(degraded) >= degraded_threshold:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "too_many_degraded_stages",
                    "message": f"{len(degraded)} stages degraded (threshold: {degraded_threshold})",
                    "stages": [m.name for m in degraded],
                }
            )

        # Check total latency
        if total_latency > latency_threshold_ms:
            alerts.append(
                {
                    "severity": "warning",
                    "type": "high_latency",
                    "message": f"Total latency {total_latency:.1f}ms exceeds {latency_threshold_ms}ms",
                    "latency_ms": total_latency,
                }
            )

        # Check error rate
        if metrics:
            errors = [m for m in metrics if m.error]
            error_rate = len(errors) / len(metrics)
            if error_rate > error_rate_threshold:
                alerts.append(
                    {
                        "severity": "critical",
                        "type": "high_error_rate",
                        "message": f"Error rate {error_rate*100:.1f}% exceeds {error_rate_threshold*100:.1f}%",
                        "error_rate": error_rate,
                    }
                )

        return alerts

    def to_dict(self) -> Dict[str, Any]:
        """Export all breaker states."""
        return {name: breaker.to_dict() for name, breaker in self._breakers.items()}
