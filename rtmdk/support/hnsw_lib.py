"""rtmdk/support/hnsw_lib.py — True HNSW index via hnswlib.

Drop-in replacement for NaiveGraphIndex with O(log N) search.
"""
from __future__ import annotations
from typing import Dict, List, Union
import numpy as np
from numpy.typing import NDArray
import logging

logger = logging.getLogger(__name__)

try:
    import hnswlib
    _HNSWLIB_AVAILABLE = True
except Exception:
    _HNSWLIB_AVAILABLE = False
    logger.info("hnswlib not available; falling back to NaiveGraphIndex.")


class HNSWLibIndex:
    """HNSW index backed by hnswlib. API-compatible with NaiveGraphIndex."""

    def __init__(self, dim: int = 64, m: int = 16, ef_construction: int = 200, space: str = "l2"):
        self.dim = dim
        self._index = hnswlib.Index(space=space, dim=dim)
        self._index.init_index(max_elements=100_000, ef_construction=ef_construction, M=m)
        self._index.set_ef(max(ef_construction // 2, 10))
        self._id_to_int: Dict[Union[int, str], int] = {}
        self._int_to_id: Dict[int, Union[int, str]] = {}
        self._next_int = 0
        self.positions: Dict[Union[int, str], np.ndarray] = {}

    def insert(self, node_id: Union[int, str], pos: NDArray):
        if node_id in self._id_to_int:
            return  # idempotent
        vec = np.asarray(pos, dtype=np.float32).reshape(1, -1)
        if vec.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vec.shape[1]}")
        idx = self._next_int
        self._next_int += 1
        self._id_to_int[node_id] = idx
        self._int_to_id[idx] = node_id
        self.positions[node_id] = vec[0]
        # Resize if needed
        current_max = self._index.max_elements
        if idx >= current_max:
            new_max = max(current_max * 2, 100_000, idx + 1)
            self._index.resize_index(new_max)
        self._index.add_items(vec, np.array([idx]))

    def remove(self, node_id: Union[int, str]):
        idx = self._id_to_int.pop(node_id, None)
        if idx is not None:
            self._int_to_id.pop(idx, None)
            self.positions.pop(node_id, None)
            self._index.mark_deleted(idx)

    def search(self, query_pos: NDArray, top_k: int = 10) -> List[Union[int, str]]:
        if not self._id_to_int:
            return []
        vec = np.asarray(query_pos, dtype=np.float32).reshape(1, -1)
        labels, _ = self._index.knn_query(vec, k=min(top_k, len(self._id_to_int)))
        result = []
        for idx in labels[0]:
            nid = self._int_to_id.get(int(idx))
            if nid is not None:
                result.append(nid)
        return result

    def set_ef(self, ef: int):
        self._index.set_ef(ef)
