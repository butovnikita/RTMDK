"""TopologyManager — tension computation, cache management, field integrity.

Extracted from RTMDKField to reduce monolithic field.py size.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField

logger = logging.getLogger(__name__)


class TopologyManager:
    """Handles tension, soft gates, pruning, and field health checks."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field

    # ------------------------------------------------------------------
    # Tension cache
    # ------------------------------------------------------------------
    def invalidate_tension_cache(self, node_id: Optional[str] = None) -> None:
        """Invalidate tension cache. If node_id given, invalidate that node and neighbors."""
        f = self.field
        dead_keys = [k for k in f._tension_cache if k not in f.nodes]
        for k in dead_keys:
            f._tension_cache.pop(k, None)

        if node_id is not None:
            f._tension_cache.pop(node_id, None)
            node = f.nodes.get(node_id)
            if node:
                for nid in list(f._tension_cache.keys()):
                    if nid == node_id:
                        continue
                    if hash(nid) % 5 == 0:
                        f._tension_cache.pop(nid, None)
        else:
            f._tension_cache.clear()

    def sweep_tension_cache(self) -> None:
        """Remove stale tension cache entries for live nodes."""
        f = self.field
        if not f._tension_cache:
            return
        if len(f._tension_cache) <= len(f.nodes) * 2:
            return
        current_step = f._step_counter
        keys_to_remove = [
            k
            for k, (tension, step) in f._tension_cache.items()
            if current_step - step > f._tension_cache_max_age * 3 and k in f.nodes
        ]
        for k in keys_to_remove:
            f._tension_cache.pop(k, None)

    # ------------------------------------------------------------------
    # Tension computation
    # ------------------------------------------------------------------
    def compute_tension(self, node_id: str, neighborhood_radius: float = 2.0) -> float:
        f = self.field
        if node_id in f._tension_cache:
            cached_tension, cached_step = f._tension_cache[node_id]
            if f._step_counter - cached_step < f._tension_cache_max_age:
                f._tension_cache_hits += 1
                return cached_tension

        f._tension_cache_misses += 1
        node = f.nodes[node_id]
        k_neighbors = 10
        neighbor_ids: List[str] = []

        if f.cfg.use_hnsw and f._index_mgr.hnsw_count() > k_neighbors:
            candidate_ids = f._index_mgr.hnsw_search(node.latent_pos, top_k=k_neighbors + 1)
            neighbor_ids = [nid for nid in candidate_ids if nid != node_id and nid in f.nodes]
        else:
            ids_to_check = f.node_index
            max_scan = 200
            if len(ids_to_check) > max_scan:
                rng = np.random.RandomState(int(hashlib.md5(node_id.encode()).hexdigest(), 16) % 2**32)
                ids_to_check = list(rng.choice(ids_to_check, size=max_scan, replace=False))

            if len(ids_to_check) < 2:
                return 0.0

            others = [(oid, f.nodes[oid]) for oid in ids_to_check if oid != node_id and oid in f.nodes]
            if not others:
                return 0.0

            other_positions = np.array([n.latent_pos for _, n in others])
            other_ids = [oid for oid, _ in others]
            dists = np.linalg.norm(other_positions - node.latent_pos, axis=1)
            within_radius = dists < neighborhood_radius
            if not np.any(within_radius):
                k = min(k_neighbors, len(dists))
                nearest_idx = np.argsort(dists)[:k]
                neighbor_ids = [other_ids[i] for i in nearest_idx]
            else:
                radius_dists = [(other_ids[i], dists[i]) for i in range(len(dists)) if within_radius[i]]
                radius_dists.sort(key=lambda x: x[1])
                neighbor_ids = [oid for oid, _ in radius_dists[:k_neighbors]]

        if len(neighbor_ids) < 2:
            tension = 0.0
        else:
            neighbors = [f.nodes[oid] for oid in neighbor_ids]
            phases = np.array([n.phase for n in neighbors])
            saliences = np.array([n.salience for n in neighbors])
            tension = 0.6 * (np.std(np.cos(phases)) + np.std(np.sin(phases))) + 0.4 * np.std(saliences)

        if f.security and not f.security.validate_tension_spike(float(tension)):
            f.stats["tension_spikes_blocked"] += 1

        result = float(tension)
        f._tension_cache[node_id] = (result, f._step_counter)
        return result

    # ------------------------------------------------------------------
    # Soft gate
    # ------------------------------------------------------------------
    def soft_gate(self, tension: float) -> float:
        f = self.field
        if not f.cfg.soft_gates:
            return 1.0
        eff = f.adaptive_threshold.get_threshold() if f.adaptive_threshold else f.cfg.tension_threshold
        return float(1 / (1 + math.exp(-(tension - eff) / f.cfg.gate_temperature)))

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    def prune_dead_nodes(self) -> None:
        f = self.field
        to_remove = [
            nid
            for nid in f.node_index
            if f.nodes[nid].amplitude < f.cfg.min_amplitude or f.nodes[nid].salience < f.cfg.min_amplitude * 0.5
        ]
        if to_remove:
            f.wal.append_delete(to_remove)
        for nid in to_remove:
            if f.cfg.use_hnsw:
                f._index_mgr.hnsw_remove(nid)
            if f.cfg.bm25_fallback:
                f._index_mgr.bm25_remove(nid)
            del f.nodes[nid]
        if to_remove:
            self.invalidate_tension_cache()
            f._cache_dirty = True
        f.node_index = [nid for nid in f.node_index if nid in f.nodes]

    # ------------------------------------------------------------------
    # Integrity check
    # ------------------------------------------------------------------
    def check_field_integrity(self) -> Dict[str, Any]:
        """Check for NaN/inf in nodes, report issues, and heal them."""
        f = self.field
        issues: List[str] = []
        n_nan = 0
        n_inf = 0
        healed: List[str] = []
        for nid, node in f.nodes.items():
            needs_heal = False
            if np.any(np.isnan(node.latent_pos)):
                n_nan += 1
                issues.append(f"NaN in {nid} — will heal")
                needs_heal = True
            if np.any(np.isinf(node.latent_pos)):
                n_inf += 1
                issues.append(f"Inf in {nid} — will heal")
                needs_heal = True
            if np.isnan(node.phase) or np.isinf(node.phase):
                issues.append(f"Invalid phase in {nid} — will heal")
                needs_heal = True
                node.phase = 0.0
            if np.isnan(node.amplitude) or node.amplitude < 0:
                issues.append(f"Invalid amplitude in {nid} — will heal")
                needs_heal = True
                node.amplitude = f.cfg.min_amplitude
            if needs_heal:
                node.latent_pos = f._rng.standard_normal(f.cfg.latent_dim).astype(np.float32) * 0.01
                healed.append(nid)
                f.stats["field_integrity_issues"] = f.stats.get("field_integrity_issues", 0) + 1
        return {
            "n_issues": len(issues),
            "n_nan": n_nan,
            "n_inf": n_inf,
            "healed": healed,
            "issues": issues[:20],
        }
