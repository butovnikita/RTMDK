"""Pipeline stages for distributed locking.

Separates distributed lock logic from the monolithic retrieve_nodes().
"""

from __future__ import annotations
from typing import Any

from rtmdk.pipeline.base import PipelineContext, PipelineStage


class DistributedLockStage(PipelineStage):
    """Acquire distributed lock before retrieval and release after.

    Must be placed at the very beginning and very end of the pipeline.
    Uses a paired release stage (DistributedLockReleaseStage) to ensure
    the lock is always released even if intermediate stages fail.
    """

    name = "distributed_lock_acquire"

    def __init__(self, lock: Any):
        self.lock = lock

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self.lock is not None:
            if not self.lock.acquire(blocking=True):
                # Log warning but continue — pipeline handles degradation
                import logging

                logging.getLogger(__name__).warning("distributed_lock_acquire: failed to acquire lock")
        return ctx


class DistributedLockReleaseStage(PipelineStage):
    """Release distributed lock at the end of the pipeline."""

    name = "distributed_lock_release"

    def __init__(self, lock: Any):
        self.lock = lock

    def process(self, ctx: PipelineContext) -> PipelineContext:
        if self.lock is not None:
            try:
                self.lock.release()
            except Exception:
                import logging

                logging.getLogger(__name__).debug(
                    "distributed_lock_release: lock already released or not held",
                    exc_info=True,
                )
        return ctx
