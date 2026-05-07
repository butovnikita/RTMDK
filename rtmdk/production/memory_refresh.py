"""
rtmdk/production/memory_refresh.py — Proactive Memory Maintenance.

Periodically boosts salience of important nodes to prevent forgetting.
Features:
- Tag nodes as "important" → periodic salience boost
- Configurable refresh interval
- Priority-based scheduling
"""

from typing import Dict


class MemoryRefresh:
    """Proactively maintains important memories.

    Usage:
        refresh = MemoryRefresh(memory, interval_steps=100)

        # Mark nodes as important:
        refresh.mark_important("node_id_123", priority=0.9)

        # Run refresh (call periodically or integrate with dreamer):
        refresh.step()
    """

    def __init__(
        self,
        memory,
        interval_steps: int = 100,
        boost_amount: float = 0.1,
        max_salience: float = 1.0,
    ):
        self.memory = memory
        self.interval_steps = interval_steps
        self.boost_amount = boost_amount
        self.max_salience = max_salience

        self._important_nodes: Dict[str, float] = {}  # node_id → priority
        self._step_counter = 0
        self._total_refreshes = 0

    def mark_important(self, node_id: str, priority: float = 0.8):
        """Mark a node as important for refresh."""
        self._important_nodes[node_id] = max(0.0, min(1.0, priority))

    def unmark_important(self, node_id: str):
        """Remove important tag from node."""
        self._important_nodes.pop(node_id, None)

    def step(self):
        """Run one refresh step. Call periodically."""
        self._step_counter += 1
        if self._step_counter % self.interval_steps != 0:
            return

        boosted = 0
        for node_id, priority in self._important_nodes.items():
            node = self.memory.field.nodes.get(node_id)
            if node is None:
                continue

            boost = self.boost_amount * priority
            node.salience = min(self.max_salience, node.salience + boost)
            node.amplitude = min(1.0, node.amplitude + boost * 0.5)
            boosted += 1

        self._total_refreshes += 1

    def get_stats(self) -> Dict:
        """Get refresh statistics."""
        return {
            "important_nodes": len(self._important_nodes),
            "total_refreshes": self._total_refreshes,
            "interval_steps": self.interval_steps,
            "step_counter": self._step_counter,
        }
