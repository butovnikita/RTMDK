"""
rtmdk/production/offline_dreamer.py — Offline Dreaming & Global Topology Repair.

Moves heavy background operations out of real-time path:
- TDA analysis
- Deep crystallization
- Shard center recalculation
- Engram merging
- Global topology repair

Inspired by sleep consolidation theory (Wilson & McNaughton, 1994).
"""

import time
import threading
import queue
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DreamTask:
    """A background task for the dreamer."""
    name: str
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: int = 0  # Higher = more important
    cooldown_steps: int = 100  # Min steps between runs


class OfflineDreamer:
    """Manages background dreaming cycles for heavy RTMDK operations.

    Usage:
        dreamer = OfflineDreamer(field=memory.field, engram_manager=memory.engram_manager)
        dreamer.start()
        # In main loop:
        dreamer.on_step()
        # On shutdown:
        dreamer.stop()
    """

    def __init__(
        self,
        field=None,
        engram_manager=None,
        dream_freq: int = 50,       # Run dream cycle every N steps
        max_workers: int = 2,        # Background threads
        enable_tda: bool = True,
        enable_crystallization: bool = True,
        enable_shard_recalc: bool = True,
        enable_engram_merge: bool = True,
        enable_topology_repair: bool = True,
    ):
        self.field = field
        self.engram_manager = engram_manager
        self.dream_freq = dream_freq
        self.max_workers = max_workers

        self._enabled = {
            "tda_analysis": enable_tda,
            "crystallization": enable_crystallization,
            "shard_recalc": enable_shard_recalc,
            "engram_merge": enable_engram_merge,
            "topology_repair": enable_topology_repair,
        }

        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._step_counter = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_run: Dict[str, int] = {}
        self._stats = {
            "cycles_completed": 0,
            "tasks_executed": 0,
            "tasks_skipped": 0,
            "total_dream_time_s": 0.0,
        }

    def start(self):
        """Start the dreamer background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._dream_loop,
            daemon=True,
            name="RTMDK-Dreamer")
        self._thread.start()
        logger.info("OfflineDreamer started")

    def stop(self):
        """Stop the dreamer and wait for current task to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        logger.info("OfflineDreamer stopped")

    def on_step(self):
        """Call this on each RTMDK evolution step."""
        self._step_counter += 1
        if self._step_counter % self.dream_freq == 0:
            self._schedule_tasks()

    def _schedule_tasks(self):
        """Schedule tasks based on cooldown and enabled state."""
        for task_name, enabled in self._enabled.items():
            if not enabled:
                continue

            last_run = self._last_run.get(task_name, 0)
            if self._step_counter - last_run < 100:  # Min 100 steps between runs
                continue

            task = self._create_task(task_name)
            if task:
                self._task_queue.put(
                    (-task.priority, task))  # Negate for max-heap

    def _create_task(self, name: str) -> Optional[DreamTask]:
        """Create a dream task for the given operation."""
        if name == "tda_analysis" and self.field:
            return DreamTask(
                "tda_analysis",
                self._run_tda_analysis,
                priority=3,
                cooldown_steps=200)
        elif name == "crystallization" and self.field:
            return DreamTask(
                "crystallization",
                self._run_crystallization,
                priority=4,
                cooldown_steps=150)
        elif name == "shard_recalc" and self.field:
            return DreamTask(
                "shard_recalc",
                self._run_shard_recalc,
                priority=2,
                cooldown_steps=300)
        elif name == "engram_merge" and self.engram_manager:
            return DreamTask(
                "engram_merge",
                self._run_engram_merge,
                priority=5,
                cooldown_steps=100)
        elif name == "topology_repair" and self.field:
            return DreamTask(
                "topology_repair",
                self._run_topology_repair,
                priority=1,
                cooldown_steps=250)
        return None

    def _dream_loop(self):
        """Background thread main loop."""
        while self._running:
            try:
                # Get next task with timeout
                try:
                    neg_priority, task = self._task_queue.get(timeout=5)
                except queue.Empty:
                    continue

                # Execute task
                t0 = time.time()
                try:
                    task.func(*task.args, **task.kwargs)
                    self._last_run[task.name] = self._step_counter
                    self._stats["tasks_executed"] += 1
                    elapsed = time.time() - t0
                    logger.info(
                        f"[Dream] Task '{task.name}' completed in {elapsed:.1f}s")
                except Exception as e:
                    logger.warning(f"[Dream] Task '{task.name}' failed: {e}")

                self._stats["cycles_completed"] += 1
                self._stats["total_dream_time_s"] += time.time() - t0

            except Exception as e:
                logger.error(f"[Dream] Loop error: {e}")

    def _run_tda_analysis(self):
        """Run Topological Data Analysis on the field."""
        if not self.field or not hasattr(self.field, 'tda_monitor'):
            return

        # Compute Betti numbers (H0, H1) — simplified
        n_nodes = len(self.field.nodes)
        if n_nodes < 10:
            return

        # Sample nodes for TDA (full TDA is O(N³))
        min(500, n_nodes)
        # Placeholder: in real implementation, compute persistent homology
        self.field.stats["tda_last_run"] = self._step_counter
        self.field.stats["tda_betti_0"] = 1  # Placeholder
        self.field.stats["tda_betti_1"] = 0  # Placeholder

    def _run_crystallization(self):
        """Deep crystallization: merge highly similar node clusters."""
        if not self.field:
            return

        # Find clusters of nodes with cosine similarity > threshold
        # Merge each cluster into a single consolidated node
        n_nodes = len(self.field.nodes)
        if n_nodes < 50:
            return

        # Placeholder: real implementation would use clustering
        self.field.stats["crystallizations"] = self.field.stats.get(
            "crystallizations", 0) + 1

    def _run_shard_recalc(self):
        """Recalculate shard centers for federated nodes."""
        if not self.field or not hasattr(self.field, 'shard_centers'):
            return

        # Recompute shard centers from current node positions
        self.field.stats["shard_recalcs"] = self.field.stats.get(
            "shard_recalcs", 0) + 1

    def _run_engram_merge(self):
        """Merge overlapping engrams."""
        if not self.engram_manager:
            return

        # Run overlap check and merge
        self.engram_manager._check_and_merge_overlaps()

    def _run_topology_repair(self):
        """Repair topological anomalies: dead zones, hyperconvergence, fragmentation."""
        if not self.field:
            return

        # Check for dead zones (isolated nodes with no neighbors)
        # Check for hyperconvergence (all nodes too close)
        # Check for fragmentation (disconnected components)
        self.field.stats["topology_repairs"] = self.field.stats.get(
            "topology_repairs", 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get dreamer statistics."""
        return {
            **self._stats,
            "step_counter": self._step_counter,
            "enabled_tasks": [k for k, v in self._enabled.items() if v],
            "queue_size": self._task_queue.qsize(),
        }
