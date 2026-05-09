"""Persistent storage for pipeline metrics.

Enables offline analysis of retrieval performance per stage.
"""
from __future__ import annotations
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class PipelineMetricsStore:
    """Append-only store for pipeline query metrics.

    Usage:
        store = PipelineMetricsStore("./pipeline_metrics.jsonl")
        store.write(ctx.to_dict())
    """

    def __init__(self, path: str, max_size_mb: float = 100.0):
        self.path = path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        dir_path = os.path.dirname(self.path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def write(self, ctx_dict: Dict[str, Any]) -> None:
        """Append a single query's metrics to the store."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **ctx_dict,
        }
        with self._lock:
            # Rotate if file too large
            if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_size_bytes:
                self._rotate()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _rotate(self) -> None:
        """Move current file to .1, .2, etc."""
        base = self.path
        for i in range(4, 0, -1):
            src = f"{base}.{i}"
            dst = f"{base}.{i + 1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        if os.path.exists(base):
            os.rename(base, f"{base}.1")

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all records from the store."""
        records = []
        files = [self.path]
        for i in range(1, 6):
            rotated = f"{self.path}.{i}"
            if os.path.exists(rotated):
                files.append(rotated)

        for filepath in files:
            if not os.path.exists(filepath):
                continue
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate statistics from stored metrics."""
        records = self.read_all()
        if not records:
            return {"queries": 0}

        total_latency = [r.get("total_latency_ms", 0) for r in records]
        stage_latencies: Dict[str, List[float]] = {}
        error_counts: Dict[str, int] = {}
        degraded_counts: Dict[str, int] = {}

        for r in records:
            for stage in r.get("stages", []):
                name = stage.get("stage", "unknown")
                latency = stage.get("latency_ms", 0)
                stage_latencies.setdefault(name, []).append(latency)
                if stage.get("error"):
                    error_counts[name] = error_counts.get(name, 0) + 1
                if stage.get("degraded"):
                    degraded_counts[name] = degraded_counts.get(name, 0) + 1

        import statistics

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {
                "count": len(values),
                "mean": round(statistics.mean(values), 3),
                "median": round(statistics.median(values), 3),
                "p95": round(sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0], 3),
                "max": round(max(values), 3),
            }

        return {
            "queries": len(records),
            "total_latency_ms": _stats(total_latency),
            "stages": {
                name: {
                    "latency_ms": _stats(latencies),
                    "errors": error_counts.get(name, 0),
                    "degraded": degraded_counts.get(name, 0),
                }
                for name, latencies in stage_latencies.items()
            },
        }
