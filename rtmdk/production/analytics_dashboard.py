"""
rtmdk/production/analytics_dashboard.py — Unified Analytics Dashboard API.

Combines MemoryAnalytics, AnalyticsEngine, and HealthMonitor into a single
dashboard interface for the REST API and UI consumption.
"""

import time
from typing import Dict, Any, Optional, List

from rtmdk.production.analytics import MemoryAnalytics
from rtmdk.production.analytics_engine import AnalyticsEngine
from rtmdk.production.health_monitor import HealthMonitor


class AnalyticsDashboard:
    """Unified dashboard for RTMDK production metrics.

    Usage:
        dashboard = AnalyticsDashboard(memory)
        overview = dashboard.get_overview()
        memory_stats = dashboard.get_memory_analytics()
        events = dashboard.get_event_series()
    """

    def __init__(
        self,
        memory: Any,
        analytics_engine: Optional[AnalyticsEngine] = None,
        health_monitor: Optional[HealthMonitor] = None,
    ):
        self.memory = memory
        self._mem_analytics = MemoryAnalytics(memory)
        self._engine = analytics_engine or AnalyticsEngine()
        self._health = health_monitor

    def get_overview(self) -> Dict[str, Any]:
        """High-level dashboard overview."""
        field = self.memory.field
        stats = field.stats if field else {}
        return {
            "timestamp": time.time(),
            "nodes": {
                "total": len(field.nodes) if field else 0,
                "max": self.memory.config.max_nodes if self.memory.config else None,
            },
            "queries": {
                "total": stats.get("total_queries", 0),
                "today": self._engine.store.query(
                    event_type="query_received",
                    since=time.time() - 86400,
                ),
            },
            "health": self._health.check_health() if self._health else None,
            "version": "8.3.0",
        }

    def get_memory_analytics(self) -> Dict[str, Any]:
        """Memory-specific analytics."""
        return {
            "topic_distribution": self._mem_analytics.get_topic_distribution(),
            "forgetting_trends": self._mem_analytics.get_forgetting_trends(),
            "retrieval_stats": self._mem_analytics.get_retrieval_stats(),
            "node_lifecycle": self._mem_analytics.get_node_lifecycle(),
        }

    def get_event_series(
        self,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Event log with optional filtering."""
        return self._engine.store.query(
            event_type=event_type,
            since=since,
            limit=limit,
        )

    def get_conversion_stats(self) -> Dict[str, Any]:
        """Conversion funnel statistics."""
        return self._engine.store.get_conversion_stats()

    def get_report(self, since: Optional[float] = None) -> Dict[str, Any]:
        """Full analytics report (last 24h by default)."""
        return self._engine.get_report(since=since)

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Pipeline-specific metrics for dashboard display.

        Returns:
            Dict with stage latencies, breaker states, error rates,
            and overall pipeline health.
        """
        if self.memory is None or not hasattr(self.memory, "build_pipeline"):
            return {"enabled": False}

        pipeline = self.memory.build_pipeline()
        stages_info = []
        open_breakers = 0

        for stage in pipeline.stages:
            breaker_state = None
            if stage.circuit_breaker is not None:
                breaker_state = stage.circuit_breaker.state.value
                if breaker_state == "open":
                    open_breakers += 1
            stages_info.append(
                {
                    "name": stage.name,
                    "enabled": stage.enabled,
                    "breaker_state": breaker_state,
                }
            )

        overall = "healthy"
        if open_breakers > 0:
            overall = "degraded"
        if open_breakers >= len(stages_info) // 2:
            overall = "unhealthy"

        return {
            "enabled": True,
            "overall": overall,
            "stages": stages_info,
            "total_stages": len(stages_info),
            "open_breakers": open_breakers,
        }

    def track_event(
        self,
        event_type: str,
        properties: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Track a custom analytics event."""
        self._engine.track(event_type, properties or {}, session_id=session_id)
