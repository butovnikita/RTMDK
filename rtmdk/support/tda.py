"""TDA monitor for RTMDK."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from rtmdk.nodes import MemoryNode


class TDAMonitor:
    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    def compute_persistence(self, nodes: Dict[str, "MemoryNode"]) -> Dict[str, Any]:
        if len(nodes) < 3:
            return {"H0": 0, "H1": 0, "avg_persistence": 0.0}
        positions = np.array([n.latent_pos for n in nodes.values()])
        n = len(positions)
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        valid = dists[dists < np.inf]
        if len(valid) < 2:
            return {"H0": n, "H1": 0, "avg_persistence": 0.0}
        threshold = np.median(valid)
        connected = [[i] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < threshold:
                    ci = cj = -1
                    for c_idx, c in enumerate(connected):
                        if i in c:
                            ci = c_idx
                        if j in c:
                            cj = c_idx
                    if ci != cj and ci >= 0 and cj >= 0:
                        connected[ci].extend(connected[cj])
                        connected.pop(cj)
        h0 = len(connected)
        h1 = max(0, len(valid) - n + h0)
        result = {"H0": h0, "H1": h1, "avg_persistence": 0.0}
        self.history.append(result)
        return result

    def get_trend(self) -> str:
        if len(self.history) < 2:
            return "stable"
        recent = [h["H1"] for h in self.history[-5:]]
        if len(recent) >= 3 and recent[-1] > recent[0] * 1.5:
            return "growing_contradictions"
        return "stable"
