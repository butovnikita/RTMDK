"""rtmdk/production/retention.py — Data retention policy manager.

Automatically prunes old memory nodes based on configurable policies.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetentionPolicy:
    """Policy for automatic data retention."""

    max_age_seconds: Optional[float] = None  # Delete nodes older than this
    max_nodes: Optional[int] = None  # Keep only N most recently accessed nodes
    enabled: bool = True


class RetentionManager:
    """Background worker that enforces retention policies.

    Usage:
        mgr = RetentionManager(memory_field)
        mgr.set_policy(RetentionPolicy(max_age_seconds=86400*30))
        mgr.start()
    """

    def __init__(self, memory_field, check_interval: float = 3600.0):
        self.field = memory_field
        self.check_interval = check_interval
        self.policy = RetentionPolicy()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._pruned_count = 0

    def set_policy(self, policy: RetentionPolicy):
        self.policy = policy

    def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def _loop(self):
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                if self.policy.enabled:
                    self._enforce()
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def _enforce(self):
        if self.field is None or not self.policy.enabled:
            return
        to_remove = []
        now = time.time()

        # Age-based pruning
        if self.policy.max_age_seconds is not None:
            cutoff = now - self.policy.max_age_seconds
            for nid, node in self.field.nodes.items():
                last_accessed = getattr(node, "last_accessed", 0) or getattr(node, "created_at", 0) or 0
                if last_accessed < cutoff:
                    to_remove.append(nid)

        # Count-based pruning (keep most recently accessed)
        if self.policy.max_nodes is not None and len(self.field.nodes) > self.policy.max_nodes:
            sorted_nodes = sorted(
                self.field.nodes.items(),
                key=lambda item: getattr(item[1], "last_accessed", 0) or getattr(item[1], "created_at", 0) or 0,
                reverse=True,
            )
            to_remove.extend(nid for nid, _ in sorted_nodes[self.policy.max_nodes:])

        if to_remove:
            self.field.delete_nodes(list(set(to_remove)))
            self._pruned_count += len(set(to_remove))

    def stats(self):
        return {
            "pruned_total": self._pruned_count,
            "policy": {
                "enabled": self.policy.enabled,
                "max_age_seconds": self.policy.max_age_seconds,
                "max_nodes": self.policy.max_nodes,
            },
        }
