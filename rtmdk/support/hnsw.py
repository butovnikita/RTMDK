"""HNSW index for RTMDK."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from numpy.typing import NDArray


class HNSWIndex:
    def __init__(self, m: int = 16, ef_construction: int = 200):
        self.m = m
        self.ef_construction = ef_construction
        self.graph: Dict[str, List[str]] = {}
        self.positions: Dict[str, NDArray] = {}

    def insert(self, node_id: str, pos: NDArray):
        self.positions[node_id] = pos
        self.graph[node_id] = []
        if len(self.positions) <= 1:
            return
        candidates = [c for c in list(self.positions.keys()) if c != node_id][:self.ef_construction]
        if candidates:
            cand_pos = np.array([self.positions[c] for c in candidates])
            dists = np.linalg.norm(cand_pos - pos, axis=1)
            nearest = [candidates[i] for i in np.argsort(dists)[:self.m]]
            self.graph[node_id] = nearest
            for nb in nearest:
                if nb in self.graph:
                    self.graph[nb].append(node_id)
                    if len(self.graph[nb]) > self.m * 2:
                        self.graph[nb] = self.graph[nb][-self.m:]

    def remove(self, node_id: str):
        self.graph.pop(node_id, None)
        self.positions.pop(node_id, None)
        for nid in self.graph:
            self.graph[nid] = [n for n in self.graph[nid] if n != node_id]

    def search(self, query_pos: NDArray, top_k: int = 10) -> List[str]:
        if not self.positions:
            return []
        start = list(self.positions.keys())[0]
        candidates = {start}
        visited = set()
        for _ in range(min(self.ef_construction, len(self.positions))):
            best = min((c for c in candidates - visited), key=lambda c: np.linalg.norm(self.positions[c] - query_pos), default=None)
            if best is None:
                break
            visited.add(best)
            candidates.update(self.graph.get(best, []))
        return sorted(candidates, key=lambda nid: np.linalg.norm(self.positions[nid] - query_pos))[:top_k]
