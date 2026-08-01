"""
rtmdk/production/analytics.py — Memory Analytics & Trends.

Provides analytical insights into memory usage patterns.
Features:
- Topic distribution over time
- Forgetting trends
- Retrieval accuracy trends
- Node lifecycle analysis
"""

import time
from typing import Dict, List, Any
from collections import defaultdict


class MemoryAnalytics:
    """Analytics engine for RTMDK memory.

    Usage:
        analytics = MemoryAnalytics(memory)

        # Get topic distribution
        topics = analytics.get_topic_distribution()

        # Get forgetting trends
        forgetting = analytics.get_forgetting_trends()

        # Get retrieval stats
        stats = analytics.get_retrieval_stats()
    """

    def __init__(self, memory):
        self.memory = memory

    def get_topic_distribution(self) -> Dict[str, int]:
        """Get distribution of nodes by topic/tier."""
        distribution: Dict[str, int] = defaultdict(int)
        for node in self.memory.field.nodes.values():
            tier = getattr(node, "tier", "unknown")
            distribution[tier] += 1
        return dict(distribution)

    def get_forgetting_trends(self) -> List[Dict[str, Any]]:
        """Analyze salience distribution to understand forgetting."""
        bins = {"high": 0, "medium": 0, "low": 0, "critical": 0}
        for node in self.memory.field.nodes.values():
            s = node.salience
            if s > 0.7:
                bins["high"] += 1
            elif s > 0.4:
                bins["medium"] += 1
            elif s > 0.1:
                bins["low"] += 1
            else:
                bins["critical"] += 1

        total = sum(bins.values()) or 1
        return [{"category": k, "count": v, "percentage": round(v / total * 100, 1)} for k, v in bins.items()]

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        stats = self.memory.field.stats
        return {
            "total_queries": stats.get("total_queries", 0),
            "bm25_fallbacks": stats.get("bm25_fallbacks", 0),
            "consolidations": stats.get("consolidations", 0),
            "engram_retrievals": stats.get("engram_retrievals", 0),
        }

    def get_node_lifecycle(self) -> Dict[str, Any]:
        """Analyze node ages and lifecycle."""
        now = time.time()
        ages = []
        for node in self.memory.field.nodes.values():
            age_hours = (now - node.created_at) / 3600
            ages.append(age_hours)

        if not ages:
            return {"count": 0}

        import numpy as np

        return {
            "count": len(ages),
            "avg_age_hours": round(np.mean(ages), 1),
            "median_age_hours": round(np.median(ages), 1),
            "min_age_hours": round(min(ages), 1),
            "max_age_hours": round(max(ages), 1),
        }

    def export_report(self) -> Dict[str, Any]:
        """Generate full analytics report."""
        return {
            "topic_distribution": self.get_topic_distribution(),
            "forgetting_trends": self.get_forgetting_trends(),
            "retrieval_stats": self.get_retrieval_stats(),
            "node_lifecycle": self.get_node_lifecycle(),
            "timestamp": time.time(),
        }
