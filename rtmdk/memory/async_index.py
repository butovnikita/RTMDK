"""
rtmdk/memory/async_index.py — Background HNSW index builder.

Defers HNSW insertions to a daemon thread that batches them periodically.
This removes the index-update bottleneck from the hot ingestion path.
"""

import threading
import time
from typing import List, Tuple, Optional, Any

import numpy as np
from numpy.typing import NDArray


class AsyncIndexBuilder:
    """Buffers HNSW inserts and flushes them in the background."""

    def __init__(
        self,
        hnsw_index: Any,
        interval_ms: int = 5000,
        batch_size: int = 1000,
        max_pending: int = 10_000,
    ):
        self.hnsw_index = hnsw_index
        self.interval_ms = interval_ms
        self.batch_size = batch_size
        self.max_pending = max_pending
        self._pending: List[Tuple[str, NDArray]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, node_id: str, latent: NDArray) -> None:
        with self._lock:
            self._pending.append((node_id, latent))
            should_flush = len(self._pending) >= self.batch_size or len(
                self._pending
            ) >= self.max_pending
        if should_flush:
            self.flush()

    def submit_batch(self, node_ids: List[str], latents: NDArray) -> None:
        with self._lock:
            for nid, latent in zip(node_ids, latents):
                self._pending.append((nid, latent))
            should_flush = len(self._pending) >= self.batch_size or len(
                self._pending
            ) >= self.max_pending
        if should_flush:
            self.flush()

    def remove(self, node_id: str) -> None:
        """Remove from pending buffer and from the underlying index."""
        with self._lock:
            self._pending = [
                (nid, lat) for nid, lat in self._pending if nid != node_id
            ]
        if hasattr(self.hnsw_index, "remove"):
            self.hnsw_index.remove(node_id)

    def flush(self) -> None:
        """Force-flush pending inserts to the HNSW index."""
        with self._lock:
            batch = self._pending
            self._pending = []
        if not batch:
            return
        nids = [nid for nid, _ in batch]
        latents = np.array([lat for _, lat in batch], dtype=np.float32)
        if hasattr(self.hnsw_index, "insert_batch"):
            self.hnsw_index.insert_batch(nids, latents)
        else:
            for nid, latent in batch:
                self.hnsw_index.insert(nid, latent)

    def close(self) -> None:
        self._stop_event.set()
        self.flush()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.interval_ms / 1000.0)
            self.flush()
