"""Event-driven scheduler and low-rank compression for RTMDK."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


class LowRankCompressor:
    """Incremental SVD-based compression of latent states."""

    def __init__(self, rank: int = 32):
        self.rank = rank
        self.U: Optional[NDArray] = None
        self.S: Optional[NDArray] = None
        self.Vt: Optional[NDArray] = None
        self._update_count = 0

    def compress(self, positions: NDArray) -> Tuple[NDArray, NDArray]:
        """Compress node positions to low-rank representation."""
        if len(positions) < 2:
            return positions, positions

        # Truncated SVD
        U, S, Vt = np.linalg.svd(positions, full_matrices=False)
        k = min(self.rank, len(S))
        self.U = U[:, :k]
        self.S = S[:k]
        self.Vt = Vt[:k, :]
        self._update_count += 1

        compressed = self.U @ np.diag(self.S)
        reconstructed = compressed @ self.Vt
        return compressed, reconstructed

    def get_compression_ratio(self, original_shape: Tuple[int, int]) -> float:
        """How much compression achieved."""
        original_size = original_shape[0] * original_shape[1]
        compressed_size = self.rank * (original_shape[0] + original_shape[1])
        return compressed_size / max(original_size, 1)

    def get_state(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "update_count": self._update_count,
            "U": self.U.tolist() if self.U is not None else None,
            "S": self.S.tolist() if self.S is not None else None,
            "Vt": self.Vt.tolist() if self.Vt is not None else None,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.rank = state.get("rank", self.rank)
        self._update_count = state.get("update_count", 0)
        if state.get("U"):
            self.U = np.array(state["U"], dtype=np.float32)
        if state.get("S"):
            self.S = np.array(state["S"], dtype=np.float32)
        if state.get("Vt"):
            self.Vt = np.array(state["Vt"], dtype=np.float32)


class EventDrivenScheduler:
    """Event-driven triggers instead of periodic step()."""

    def __init__(self) -> None:
        self._event_queue: deque[Dict[str, Any]] = deque(maxlen=1000)
        self._event_counts: Dict[str, int] = defaultdict(int)

    def enqueue(self, event_type: str, payload: Dict[str, Any]):
        self._event_queue.append({
            "type": event_type,
            "payload": payload,
            "timestamp": time.time(),
        })
        self._event_counts[event_type] += 1

    def process_pending(self, field: Any, max_events: int = 10) -> int:
        """Process pending events."""
        processed = 0
        while self._event_queue and processed < max_events:
            event = self._event_queue.popleft()
            etype = event["type"]
            event["payload"]

            if etype == "node_added":
                pass  # Already handled by add_node
            elif etype == "high_tension":
                field.consolidate()
                processed += 1
            elif etype == "query":
                pass  # Already handled by query
            elif etype == "crystallize":
                if hasattr(field, '_crystallize_recurring'):
                    field._crystallize_recurring()
                processed += 1
            elif etype == "compress":
                if hasattr(field, '_compress_field'):
                    field._compress_field()
                processed += 1

        return processed

    def get_stats(self) -> Dict[str, Any]:
        return {
            "queue_depth": len(self._event_queue),
            "event_counts": dict(self._event_counts),
        }
