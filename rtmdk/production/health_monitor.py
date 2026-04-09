"""
rtmdk/production/health_monitor.py — Health Checks & Real-Time Monitoring.

Provides health checks, metrics, and alerting for RTMDK.
Features:
- Health check: /health endpoint equivalent
- Real-time metrics: node count, latency, RAM, cache hit rate
- Alert callbacks when thresholds exceeded
- Prometheus-style metrics export
"""

import os
import time
import tracemalloc
from typing import Dict, List, Optional, Any, Callable


class HealthMonitor:
    """Monitors RTMDK health and provides metrics.
    
    Usage:
        monitor = HealthMonitor(memory)
        
        # Check health
        health = monitor.check_health()
        
        # Get metrics
        metrics = monitor.get_metrics()
        
        # Set up alerts
        monitor.add_alert("high_latency", threshold=500, callback=my_callback)
    """
    
    def __init__(self, memory, check_interval: int = 60):
        self.memory = memory
        self.check_interval = check_interval
        self._alerts: List[Dict] = []
        self._last_check = 0
        self._latency_history: List[float] = []
        self._last_health = None
    
    def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check.
        
        Returns:
            {"status": "healthy"|"degraded"|"unhealthy", "checks": {...}}
        """
        t0 = time.time()
        checks = {}
        status = "healthy"
        
        # Check node count
        node_count = len(self.memory.field.nodes)
        max_nodes = self.memory.config.max_nodes or 100000
        node_ratio = node_count / max_nodes
        checks["node_count"] = {"value": node_count, "max": max_nodes, "ratio": round(node_ratio, 3)}
        if node_ratio > 0.9:
            status = "degraded"
            checks["node_count"]["warning"] = "Approaching max_nodes limit"
        
        # Check field stability
        stats = self.memory.field.stats
        checks["field_stats"] = {
            "total_queries": stats.get("total_queries", 0),
            "consolidations": stats.get("consolidations", 0),
            "bm25_fallbacks": stats.get("bm25_fallbacks", 0),
        }
        
        # Check RAM
        try:
            current, peak = tracemalloc.get_traced_memory()
            checks["memory"] = {
                "current_mb": round(current / 1024 / 1024, 1),
                "peak_mb": round(peak / 1024 / 1024, 1),
            }
        except Exception:
            checks["memory"] = {"status": "unavailable"}
        
        # Check latency
        if self._latency_history:
            avg_latency = sum(self._latency_history[-100:]) / min(len(self._latency_history), 100)
            checks["latency"] = {"avg_ms": round(avg_latency, 1)}
            if avg_latency > 500:
                status = "degraded"
                checks["latency"]["warning"] = "High average latency"
        
        # Check for NaN/Inf in nodes (integrity)
        nan_count = 0
        for node in list(self.memory.field.nodes.values())[:1000]:
            if hasattr(node, 'latent_pos'):
                import numpy as np
                if np.any(np.isnan(node.latent_pos)) or np.any(np.isinf(node.latent_pos)):
                    nan_count += 1
        checks["integrity"] = {"nan_inf_nodes": nan_count}
        if nan_count > 0:
            status = "unhealthy"
        
        self._last_health = {
            "status": status,
            "timestamp": time.time(),
            "checks": checks,
        }
        
        # Fire alerts
        self._fire_alerts(status, checks)
        
        return self._last_health
    
    def add_alert(self, name: str, threshold: float, callback: Callable):
        """Add an alert callback."""
        self._alerts.append({
            "name": name,
            "threshold": threshold,
            "callback": callback,
        })
    
    def record_latency(self, latency_ms: float):
        """Record a latency measurement."""
        self._latency_history.append(latency_ms)
        if len(self._latency_history) > 10000:
            self._latency_history = self._latency_history[-5000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Prometheus-style metrics."""
        health = self._last_health or self.check_health()
        
        return {
            "rtmdk_nodes_total": len(self.memory.field.nodes),
            "rtmdk_health_status": health["status"],
            "rtmdk_total_queries": health["checks"].get("field_stats", {}).get("total_queries", 0),
            "rtmdk_consolidations": health["checks"].get("field_stats", {}).get("consolidations", 0),
            "rtmdk_memory_mb": health["checks"].get("memory", {}).get("current_mb", 0),
            "rtmdk_avg_latency_ms": health["checks"].get("latency", {}).get("avg_ms", 0),
            "rtmdk_integrity_issues": health["checks"].get("integrity", {}).get("nan_inf_nodes", 0),
        }
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        metrics = self.get_metrics()
        lines = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"# TYPE {name} gauge\n{name} {value}")
        return '\n'.join(lines)
    
    def _fire_alerts(self, status: str, checks: Dict):
        """Fire alert callbacks."""
        for alert in self._alerts:
            if status == "unhealthy" or "warning" in str(checks):
                try:
                    alert["callback"](alert["name"], status, checks)
                except Exception:
                    pass
