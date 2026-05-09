"""AsyncPipelineManager — background worker lifecycle for async pipeline.

Extracted from RTMDKField to reduce monolithic field.py size.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField

logger = logging.getLogger(__name__)


class AsyncPipelineManager:
    """Manages background async workers for field evolution and batch save."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field

    async def start_workers(self) -> None:
        """Start background worker tasks for async pipeline with lifecycle tracking."""
        f = self.field
        if f._workers_started:
            return
        f._workers_started = True
        t_evolve = asyncio.create_task(self._worker_evolve())
        t_save = asyncio.create_task(self._worker_save())
        f._workers.extend([t_evolve, t_save])

    async def _worker_evolve(self) -> None:
        """Background worker for field evolution with throttling."""
        f = self.field
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(f.evolve_q.get(), timeout=1.0)
                    inputs = payload.get("inputs", {})

                    backpressure_ok = f._backpressure_events < 3

                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, f.step, inputs)

                    f._last_successful_step = time.time()

                    if backpressure_ok and f.meta_controller:
                        if f.meta_controller.should_optimize():
                            f._circuit_breakers["MetaControllerOptimize"].call(
                                f.meta_controller.optimize, f)

                    if f._backpressure_events > 0:
                        f._backpressure_events = max(0, f._backpressure_events - 1)
                        if f._backpressure_events == 0 and f._heavy_modules_degraded:
                            f._heavy_modules_degraded = False
                            f.stats["backpressure_degraded_mode"] = f.stats.get(
                                "backpressure_degraded_mode", 0) + 1
                            logger.info(
                                "Backpressure recovered — heavy modules re-enabled")
                        if f._backpressure_events == 0:
                            f.stats["last_backpressure_recovery"] = time.time()

                    f.evolve_q.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    f._backpressure_events += 1
                    logger.exception("Evolve worker error")
        except asyncio.CancelledError:
            logger.info("Evolve worker cancelled cleanly.")

    async def _worker_save(self) -> None:
        """Background worker for batch ingestion."""
        f = self.field
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(f.save_q.get(), timeout=1.0)
                    embeddings = payload.get("embeddings")
                    contents = payload.get("contents")
                    modalities = payload.get("modalities")
                    if embeddings is not None and contents is not None:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            f.add_nodes_batch,
                            embeddings,
                            contents,
                            None,  # phases
                            None,  # node_ids
                            None,  # session_ids
                            modalities,
                            False,  # skip_projection
                        )
                    self._track_queue_depth()
                    f.save_q.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    logger.exception("Save worker error")
        except asyncio.CancelledError:
            logger.info("Save worker cancelled cleanly.")

    def _track_queue_depth(self) -> None:
        """Track async queue depths for monitoring."""
        f = self.field
        if f.cfg.async_pipeline and f.evolve_q:
            f.stats["async_queue_depth"] = (
                f.evolve_q.qsize() +
                (f.save_q.qsize() if f.save_q else 0) +
                (f.query_q.qsize() if f.query_q else 0)
            )
