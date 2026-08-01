"""NodeCacheManager — pre-materialized numpy arrays for vectorized resonance."""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Any

import numpy as np
from numpy.typing import NDArray


class NodeCacheManager:
    """Manages cached numpy arrays built from field nodes.

    Encapsulates _build_node_cache logic and the associated arrays,
    eliminating O(N) Python dict lookups on every query.
    """

    def __init__(self):
        self._cached_positions: Optional[NDArray] = None  # (N, D)
        self._cached_scales: Optional[NDArray] = None  # (N,) — int8 only
        self._cached_norms_sq: Optional[NDArray] = None  # (N,) — precomputed ||p||^2
        self._cached_phases: Optional[NDArray] = None  # (N,)
        self._cached_amplitudes: Optional[NDArray] = None  # (N,)
        self._cached_saliences: Optional[NDArray] = None  # (N,)
        self._cached_modal_weights: Optional[NDArray] = None  # (N,)
        self._cached_gates: Optional[NDArray] = None  # (N,)
        self._cached_causal_boost: Optional[NDArray] = None  # (N,)
        self._cache_dirty: bool = False
        self._node_id_to_cached_idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Properties (read-only for consumers)
    # ------------------------------------------------------------------
    @property
    def positions(self) -> Optional[NDArray]:
        return self._cached_positions

    @property
    def phases(self) -> Optional[NDArray]:
        return self._cached_phases

    @property
    def amplitudes(self) -> Optional[NDArray]:
        return self._cached_amplitudes

    @property
    def saliences(self) -> Optional[NDArray]:
        return self._cached_saliences

    @property
    def modal_weights(self) -> Optional[NDArray]:
        return self._cached_modal_weights

    @property
    def gates(self) -> Optional[NDArray]:
        return self._cached_gates

    @property
    def causal_boost(self) -> Optional[NDArray]:
        return self._cached_causal_boost

    @property
    def scales(self) -> Optional[NDArray]:
        """Per-node latent_scale for int8 fast path."""
        return self._cached_scales

    @property
    def norms_sq(self) -> Optional[NDArray]:
        """Precomputed squared norms of cached positions."""
        return self._cached_norms_sq

    @property
    def dirty(self) -> bool:
        return self._cache_dirty

    @property
    def node_id_to_idx(self) -> Dict[str, int]:
        return self._node_id_to_cached_idx

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def invalidate(self) -> None:
        self._cache_dirty = True

    def clear(self) -> None:
        self._cached_positions = None
        self._cached_scales = None
        self._cached_norms_sq = None
        self._cached_phases = None
        self._cached_amplitudes = None
        self._cached_saliences = None
        self._cached_modal_weights = None
        self._cached_gates = None
        self._cached_causal_boost = None
        self._cache_dirty = False
        self._node_id_to_cached_idx.clear()

    def build(self, field: Any) -> None:
        """Rebuild all cached arrays from the field's current nodes.

        Parameters
        ----------
        field : RTMDKField
            Must expose: nodes, node_index, cfg, _quant, _tiered_store,
            causal_engine.

        Thread-safety: the whole rebuild runs under ``field._write_lock``
        (RLock) so concurrent add/delete writers cannot mutate the node
        dict mid-iteration or interleave a second rebuild between the
        per-attribute assignments (torn-read fix, 2026-08-01).
        Field-like objects without a lock (test doubles) run unlocked.
        """
        lock = getattr(field, "_write_lock", None)
        with lock if lock is not None else contextlib.nullcontext():
            self._build_locked(field)

    def _build_locked(self, field: Any) -> None:
        """Actual rebuild; must be called with ``field._write_lock`` held."""
        # Thread-safety: compact node_index to exclude deleted nodes
        if field._tiered_store is not None:
            valid_entries = list(field._tiered_store.cacheable_nodes())
        else:
            valid_entries = [(nid, field.nodes[nid]) for nid in field.node_index if nid in field.nodes]
        n = len(valid_entries)
        is_int8 = field._quant.mode == "int8"
        cache_dtype = np.int8 if is_int8 else field._quant.dtype
        if n == 0:
            self._cached_positions = np.empty((0, field.cfg.latent_dim), dtype=cache_dtype)
            self._cached_scales = np.empty(0, dtype=np.float32) if is_int8 else None
            self._cached_phases = np.empty(0, dtype=np.float32)
            self._cached_amplitudes = np.empty(0, dtype=np.float32)
            self._cached_saliences = np.empty(0, dtype=np.float32)
            self._cached_modal_weights = np.empty(0, dtype=np.float32)
            self._cached_gates = np.empty(0, dtype=np.float32)
            self._cached_causal_boost = np.empty(0, dtype=np.float32)
            self._cache_dirty = False
            field.node_index = []
            return

        positions: NDArray = np.zeros((n, field.cfg.latent_dim), dtype=cache_dtype)
        scales: Optional[NDArray] = np.ones(n, dtype=np.float32) if is_int8 else None
        phases: NDArray = np.zeros(n, dtype=np.float32)
        amplitudes: NDArray = np.zeros(n, dtype=np.float32)
        saliences: NDArray = np.zeros(n, dtype=np.float32)
        modal_weights: NDArray = np.zeros(n, dtype=np.float32)
        gates: NDArray = np.ones(n, dtype=np.float32)
        causal_boost: NDArray = np.zeros(n, dtype=np.float32)

        for i, (nid, node) in enumerate(valid_entries):
            if is_int8:
                assert scales is not None  # scales array is created when is_int8
                positions[i] = node.latent_pos
                scales[i] = getattr(node, "latent_scale", 1.0)
            else:
                positions[i] = field._quant.dequantize(
                    node.latent_pos,
                    getattr(node, "latent_scale", 1.0),
                    getattr(node, "latent_zero_point", 0.0),
                )
            phases[i] = node.phase
            amplitudes[i] = node.amplitude
            saliences[i] = node.salience
            modal_weights[i] = node.modal_weight
            if field.cfg.soft_gates and hasattr(node, "soft_gate"):
                gates[i] = node.soft_gate
            if field.causal_engine and hasattr(node, "causal_parents") and node.causal_parents:
                cb = sum(node.causal_strength.get(p, 0) for p in node.causal_parents)
                causal_boost[i] = 1.0 + 0.1 * cb

        self._cached_positions = positions
        self._cached_scales = scales
        self._cached_norms_sq = np.einsum("ij,ij->i", positions.astype(np.float32), positions.astype(np.float32))
        self._cached_phases = phases
        self._cached_amplitudes = amplitudes
        self._cached_saliences = saliences
        self._cached_modal_weights = modal_weights
        self._cached_gates = gates
        self._cached_causal_boost = causal_boost
        self._cache_dirty = False
        field.node_index = [nid for nid, _ in valid_entries]
        self._node_id_to_cached_idx = {nid: i for i, (nid, _) in enumerate(valid_entries)}

    def ensure_built(self, field: Any) -> None:
        """Lazy rebuild if dirty or missing."""
        if self._cache_dirty or self._cached_positions is None:
            self.build(field)

    def get_indices(self, node_ids: List[str]) -> NDArray:
        """Map node IDs to cached array indices."""
        mapping = self._node_id_to_cached_idx
        return np.array([mapping[nid] for nid in node_ids if nid in mapping], dtype=np.int32)
