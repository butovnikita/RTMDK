"""BatchResonanceEngine — R10.2 split from QueryManager God object.

R10.2 (2026-08-24, audit/risks-2026-08-24): QueryManager (853 lines) and
NodeManager (660 lines) were new god modules after Leadership Cleanup.
This file extracts batch resonance (vectorized resonance across many queries)
so QueryManager can delegate. NodeCacheManager already exists
(cache_manager.py) — this complements it. See docs/RISKS.md R10.2.

Currently a thin wrapper around ResonanceEngine.batch_response_* for
testability and future DI; QueryManager will migrate incrementally.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from rtmdk.memory.resonance import ResonanceEngine


class BatchResonanceEngine:
    """Vectorized resonance for batches — extracted from QueryManager."""

    def __init__(self, resonance_engine: ResonanceEngine):
        self._engine = resonance_engine

    def batch_response_numpy(
        self,
        query_latents: NDArray,
        query_phases: NDArray,
        node_positions: NDArray,
        node_phases: NDArray,
        node_amplitudes: NDArray,
        node_saliences: NDArray,
    ) -> NDArray:
        return self._engine.batch_response_numpy(
            query_latents, query_phases, node_positions, node_phases, node_amplitudes, node_saliences
        )

    def batch_response_for_ids(
        self,
        query_latents: NDArray,
        query_phases: NDArray,
        node_ids: List[str],
        field,  # FieldLike
    ) -> NDArray:
        """Batch resonance for a list of node_ids via field snapshot (R4.1)."""
        # Snapshot under lock is done by caller (QueryManager); this is pure compute.
        return self.batch_response_numpy(
            query_latents,
            query_phases,
            np.array([field.nodes[nid].latent_pos for nid in node_ids]),
            np.array([field.nodes[nid].phase for nid in node_ids]),
            np.array([field.nodes[nid].amplitude for nid in node_ids]),
            np.array([field.nodes[nid].salience for nid in node_ids]),
        )
