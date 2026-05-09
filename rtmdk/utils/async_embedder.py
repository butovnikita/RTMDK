"""Async embedder wrapper with request batching.

Wraps a synchronous embedder to provide async interface with
automatic request coalescing for better throughput under load.
"""
from __future__ import annotations
import asyncio
import logging
from typing import List, Callable
import numpy as np

logger = logging.getLogger(__name__)


class AsyncEmbedder:
    """Async wrapper around a synchronous embedder function.

    Batches concurrent requests to reduce total latency.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray],
        batch_size: int = 16,
        max_wait_ms: float = 10.0,
    ):
        self.embed_fn = embed_fn
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: List[asyncio.Future] = []
        self._texts: List[str] = []
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    async def embed(self, text: str) -> np.ndarray:
        """Embed text asynchronously."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        async with self._lock:
            self._queue.append(future)
            self._texts.append(text)
            should_flush = len(self._queue) >= self.batch_size
            if should_flush and self._task is not None:
                self._task.cancel()
                self._task = None
        if should_flush:
            await self._flush()
        else:
            async with self._lock:
                if self._task is None:
                    self._task = asyncio.create_task(self._delayed_flush())
        return await future

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts asynchronously."""
        return await asyncio.gather(*[self.embed(t) for t in texts])

    async def _delayed_flush(self) -> None:
        await asyncio.sleep(self.max_wait_ms / 1000.0)
        await self._flush()

    async def _flush(self) -> None:
        async with self._lock:
            if not self._queue:
                return
            futures = self._queue[:self.batch_size]
            texts = self._texts[:self.batch_size]
            self._queue = self._queue[self.batch_size:]
            self._texts = self._texts[self.batch_size:]
            self._task = None

        try:
            # Run blocking embed in thread pool
            loop = asyncio.get_running_loop()
            embs = await loop.run_in_executor(None, lambda: [self.embed_fn(t) for t in texts])
            for fut, emb in zip(futures, embs):
                if not fut.done():
                    fut.set_result(emb)
        except Exception as e:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)
