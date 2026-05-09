"""ConsolidationManager — spectral + pairwise node merging.

Extracted from RTMDKField to reduce monolithic field.py size.
Delegates back to the parent field for node access, indexing,
kalman filtering, versioning, etc.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import numpy as np

from rtmdk.memory.config import ConsolidationMode
from rtmdk.memory.geometry import exp_map_poincare, log_map_poincare, poincare_midpoint
from rtmdk.memory.spectral import spectral_cluster_nodes
from rtmdk.support.version_control import NodeDelta

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField


class ConsolidationManager:
    """Handles node consolidation (merging) and post-merge bookkeeping."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def consolidate(self, mode: Optional[ConsolidationMode] = None) -> List[str]:
        """Run one consolidation pass over the field.

        Returns list of updated (merged) node IDs.
        """
        field = self.field
        cfg = field.cfg
        mode = mode or cfg.consolidation_mode
        updated: List[str] = []
        eff_threshold = field.get_effective_threshold()

        pre_state: Dict = {}
        if cfg.enable_rollback or cfg.self_sup_verify_after_consolidate:
            for nid in field.node_index:
                n = field.nodes[nid]
                pre_state[nid] = {
                    "latent_pos": n.latent_pos.copy(),
                    "phase": n.phase,
                    "amplitude": n.amplitude,
                    "salience": n.salience,
                    "tension": n.tension,
                    "soft_gate": n.soft_gate,
                    "content": dict(n.content),
                    "lineage": list(n.lineage),
                    "causal_strength": dict(n.causal_strength),
                    "causal_parents": list(n.causal_parents),
                }

        # Snapshot node_index to avoid mutation issues
        node_index_snapshot = list(field.node_index)
        for nid in node_index_snapshot:
            if nid not in field.nodes:
                continue
            tension = field._compute_tension(nid)
            field.nodes[nid].tension = tension
            field.nodes[nid].soft_gate = field._soft_gate(tension)
            if field.adaptive_threshold:
                field.adaptive_threshold.record_tension(tension)
                field.stats["adaptive_threshold_value"] = field.adaptive_threshold.get_threshold()

        high_tension = [
            nid for nid in node_index_snapshot
            if nid in field.nodes and field.nodes[nid].tension > eff_threshold
        ]
        processed: Set[str] = set()
        pending_deletions: List[str] = []
        node_index_snapshot = list(field.node_index)
        n_snap = len(node_index_snapshot)

        # Spectral clustering (optional)
        if cfg.spectral_consolidation and len(high_tension) >= 3:
            spectral_merged = self._spectral_merge_clusters(
                high_tension, mode, updated, pending_deletions, processed, pre_state
            )
            if spectral_merged:
                high_tension = [
                    nid for nid in high_tension
                    if nid not in processed and nid in field.nodes
                ]

        # HNSW or vectorized pairwise merge
        if cfg.use_hnsw and field.hnsw_index and n_snap > getattr(cfg, "hnsw_min_nodes", 50):
            if n_snap <= 50:
                field.stats["hnsw_bypassed"] = field.stats.get("hnsw_bypassed", 0) + 1
            for nid in high_tension:
                if nid in processed or nid not in field.nodes:
                    continue
                node = field.nodes[nid]
                candidate_ids = field._index_mgr.hnsw_search(
                    node.latent_pos, top_k=min(50, n_snap)
                )
                candidates = []
                for oid in candidate_ids:
                    if oid == nid or oid in processed or oid not in field.nodes:
                        continue
                    other = field.nodes[oid]
                    dist = np.linalg.norm(node.latent_pos - other.latent_pos)
                    if dist >= 2.5:
                        continue
                    pd = min(abs(node.phase - other.phase), 2 * np.pi - abs(node.phase - other.phase))
                    if pd > 1.0:
                        candidates.append((oid, dist, pd))
                if not candidates:
                    continue
                candidates.sort(key=lambda x: x[1])
                pid = candidates[0][0]
                if pid not in field.nodes:
                    continue
                self._do_merge(nid, pid, mode, updated, pending_deletions, processed, pre_state)
        else:
            # Vectorized fallback
            snap_positions = np.array([
                field.nodes[oid].latent_pos for oid in node_index_snapshot if oid in field.nodes
            ])
            snap_ids = [oid for oid in node_index_snapshot if oid in field.nodes]
            snap_phases = np.array([field.nodes[oid].phase for oid in snap_ids])
            snap_id_to_idx = {nid: idx for idx, nid in enumerate(snap_ids)}

            for nid in high_tension:
                if nid in processed or nid not in field.nodes:
                    continue
                node_idx = snap_id_to_idx.get(nid)
                if node_idx is None:
                    continue
                node_pos = snap_positions[node_idx]
                dists = np.linalg.norm(snap_positions - node_pos, axis=1)
                phase_diffs = np.minimum(
                    np.abs(snap_phases - field.nodes[nid].phase),
                    2 * np.pi - np.abs(snap_phases - field.nodes[nid].phase)
                )
                mask = (dists < 2.5) & (phase_diffs > 1.0)
                candidate_indices = np.where(mask)[0]
                if len(candidate_indices) == 0:
                    continue
                sorted_indices = candidate_indices[np.argsort(dists[candidate_indices])]
                pid = snap_ids[sorted_indices[0]]
                if pid not in field.nodes or pid in processed:
                    continue
                self._do_merge(nid, pid, mode, updated, pending_deletions, processed, pre_state)

        # Apply deletions
        if pending_deletions:
            field.wal.append_delete(pending_deletions)
        for pid in pending_deletions:
            if pid in field.nodes:
                del field.nodes[pid]
        if pending_deletions:
            field._invalidate_tension_cache()
        field._sweep_tension_cache()
        field.node_index = [nid for nid in field.node_index if nid in field.nodes]
        if pending_deletions or updated:
            field._cache_dirty = True

        if updated:
            verify_limit = min(10, len(updated))
            self._verify_consistency(updated[:verify_limit], pre_state)

        field._prune_dead_nodes()
        field.stats["active_nodes"] = len(field.nodes)

        if pre_state and updated:
            scores = []
            for nid in updated:
                if nid in field.nodes and nid in pre_state:
                    o, n = pre_state[nid]["latent_pos"], field.nodes[nid].latent_pos
                    scores.append(max(0, np.dot(o, n) / (np.linalg.norm(o) * np.linalg.norm(n) + 1e-8)))
            if scores:
                field._stability_buffer.append(np.mean(scores))
                field.stats["field_stability"] = float(np.mean(field._stability_buffer))

        if cfg.enable_rollback and pre_state:
            field._rollback_history.append({
                "timestamp": time.time(), "pre_state": pre_state, "updated": updated
            })
            if len(field._rollback_history) > cfg.max_rollback_history:
                field._rollback_history.pop(0)

        if field.version_control and updated:
            deltas = []
            for nid in updated:
                if nid in field.nodes:
                    deltas.append(NodeDelta(
                        node_id=nid, action="merged",
                        old_state=pre_state.get(nid),
                        new_state=field.nodes[nid].to_dict()
                    ))
            for pid in pending_deletions:
                if pid in pre_state:
                    deltas.append(NodeDelta(
                        node_id=pid, action="deleted",
                        old_state=pre_state.get(pid)
                    ))
            if deltas:
                field.version_control.create_version(
                    deltas,
                    message=f"consolidation: {len(updated)} merged, {len(pending_deletions)} deleted"
                )
                field.stats["current_version"] = field.version_control.current_version
                field.stats["n_versions"] = field.version_control.n_versions

        if field.role_router and updated:
            affected_roles: Set[str] = set()
            for nid in updated:
                role = field.role_router.get_node_role(nid)
                if role in field.role_router.shards:
                    field.role_router.shards[role].n_consolidations += 1
                    affected_roles.add(role)

        if updated:
            field._cache_dirty = True

        if field.learned_consolidator is not None and field.stats["consolidations"] % 20 == 0 and field.stats["consolidations"] > 0:
            field._train_learned_consolidator()

        field.wal.append_consolidate(updated)
        field._dirty = True
        return updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _do_merge(
        self,
        nid: str,
        pid: str,
        mode: ConsolidationMode,
        updated: List[str],
        pending_deletions: List[str],
        processed: Set[str],
        pre_state: Dict,
    ) -> bool:
        """Execute a single merge of nid with pid."""
        field = self.field
        cfg = field.cfg
        node = field.nodes[nid]
        partner = field.nodes[pid]

        if cfg.do_calculus_validation and field.causal_engine:
            validation = field.causal_engine.validate_consolidation(nid, pid)
            field.stats["consolidation_validations"] += 1
            if not validation["safe"]:
                field.stats["blocked_consolidations"] += 1
                processed.add(nid)
                processed.add(pid)
                return False

        if cfg.domain_consolidation_guard and node.domain != partner.domain:
            if partner.id not in node.conflict_with:
                node.conflict_with.append(partner.id)
            if node.id not in partner.conflict_with:
                partner.conflict_with.append(node.id)
            processed.add(nid)
            processed.add(pid)
            return False

        gate = field._soft_gate(max(node.tension, partner.tension))

        if cfg.enable_rollback:
            node.pre_consolidation_pos = node.latent_pos.copy()

        if field.diff_consolidation and mode == ConsolidationMode.DIALECTICAL:
            synth = field.diff_consolidation.compute_synthesis(node, partner, gate)
            if cfg.hyperbolic:
                node.latent_pos = exp_map_poincare(
                    log_map_poincare(synth["latent_pos"], node.latent_pos, cfg.ball_radius),
                    node.latent_pos, cfg.ball_radius,
                )
            else:
                node.latent_pos = synth["latent_pos"]
            node.phase = synth["phase"]
            node.amplitude = synth["amplitude"]
            node.salience = synth["salience"]
        elif mode == ConsolidationMode.DIALECTICAL:
            if cfg.hyperbolic:
                node.latent_pos = poincare_midpoint(node.latent_pos, partner.latent_pos, cfg.ball_radius)
            else:
                field._merge_latents(node, partner)
            node.phase = np.arctan2(
                0.5 * (np.sin(node.phase) + np.sin(partner.phase)),
                0.5 * (np.cos(node.phase) + np.cos(partner.phase))
            ) % (2 * np.pi)
            node.amplitude = min(1.0, 0.8 * (node.amplitude + partner.amplitude))
            node.salience = min(1.0, 0.7 * (node.salience + partner.salience))
        else:
            if cfg.hyperbolic:
                node.latent_pos = poincare_midpoint(node.latent_pos, partner.latent_pos, cfg.ball_radius)
            else:
                field._merge_latents(node, partner)
            node.phase = np.arctan2(
                0.5 * (np.sin(node.phase) + np.sin(partner.phase)),
                0.5 * (np.cos(node.phase) + np.cos(partner.phase))
            ) % (2 * np.pi)

        if field.kalman_filter is not None:
            if node.covariance is not None:
                node.covariance = field.kalman_filter.predict(node.covariance)
            if node.covariance is not None and partner.covariance is not None:
                _, node.covariance = field.kalman_filter.update(
                    node.latent_pos, partner.latent_pos, node.covariance
                )
                node.covariance = field.kalman_filter.merge_covariance(
                    node.covariance, partner.covariance
                )
            elif partner.covariance is not None:
                node.covariance = partner.covariance.copy()

        node.tension = 0.0
        node.soft_gate = 1.0
        node.lineage = [f"{node.id}+{pid}"] + node.lineage + partner.lineage
        node.content["synthesis_note"] = f"Consolidated with {pid} at t={time.time():.0f}"
        if "merged_content" not in node.content:
            node.content["merged_content"] = []
        node.content["merged_content"].append(field._extract_text(partner.content))

        if field.causal_engine:
            for parent, strength in partner.causal_strength.items():
                if parent not in node.causal_strength:
                    node.causal_strength[parent] = strength
                else:
                    node.causal_strength[parent] = max(node.causal_strength[parent], strength)

        if cfg.use_hnsw:
            field._index_mgr.hnsw_remove(pid)
            field._index_mgr.hnsw_insert(nid, node.latent_pos)
        if cfg.bm25_fallback:
            field._index_mgr.bm25_remove(pid)

        pending_deletions.append(pid)
        processed.add(pid)
        updated.append(nid)
        field.stats["consolidations"] += 1
        processed.add(nid)
        return True

    def _verify_consistency(
        self,
        updated_nodes: List[str],
        pre_state: Optional[Dict] = None,
    ) -> None:
        """Self-supervision: probe merged nodes to verify they remain retrievable."""
        field = self.field
        from collections import deque
        if not isinstance(field._stability_buffer, deque):
            field._stability_buffer = deque(field._stability_buffer, maxlen=100)

        for nid in updated_nodes:
            if nid not in field.nodes:
                continue
            node = field.nodes[nid]
            probe = node.latent_pos + field._rng.normal(0, 0.05, node.latent_pos.shape)
            results = field.query(probe, phase=node.phase, top_k=1)
            if results and results[0][0] == nid:
                node.self_sup_score = max(0.5, results[0][1])
            else:
                node.self_sup_score *= 0.9

    def _spectral_merge_clusters(
        self,
        high_tension: List[str],
        mode: ConsolidationMode,
        updated: List[str],
        pending_deletions: List[str],
        processed: Set[str],
        pre_state: Dict,
    ) -> bool:
        """Spectral Graph Laplacian clustering for consolidation."""
        field = self.field
        if len(high_tension) < 3:
            return False

        positions = []
        phases = []
        valid_nids = []
        for nid in high_tension:
            if nid in field.nodes and nid not in processed:
                node = field.nodes[nid]
                positions.append(node.latent_pos)
                phases.append(node.phase)
                valid_nids.append(nid)

        if len(valid_nids) < 3:
            return False

        clusters = spectral_cluster_nodes(
            np.array(positions, dtype=np.float32),
            np.array(phases, dtype=np.float32),
            max_clusters=field.cfg.spectral_max_clusters,
            sigma=field.cfg.spectral_sigma,
            timeout_ms=500.0,
            rng=field._rng,
        )
        if clusters is None:
            return False

        merged_any = False
        for cluster_indices in clusters.values():
            if len(cluster_indices) < 2:
                continue
            cluster_nids = [valid_nids[i] for i in cluster_indices]
            while len(cluster_nids) >= 2:
                best_pair = None
                best_dist = float("inf")
                cluster_positions = {
                    nid: field.nodes[nid].latent_pos for nid in cluster_nids if nid in field.nodes
                }
                if len(cluster_positions) < 2:
                    break
                nids_list = list(cluster_positions.keys())
                for i in range(len(nids_list)):
                    for j in range(i + 1, len(nids_list)):
                        nid_i, nid_j = nids_list[i], nids_list[j]
                        d = np.linalg.norm(cluster_positions[nid_i] - cluster_positions[nid_j])
                        if d < best_dist:
                            best_dist = d
                            best_pair = (nid_i, nid_j)
                if best_pair is None:
                    break
                nid, pid = best_pair
                if nid not in field.nodes or pid not in field.nodes or nid in processed or pid in processed:
                    break
                if self._do_merge(nid, pid, mode, updated, pending_deletions, processed, pre_state):
                    merged_any = True
                    cluster_nids = [n for n in cluster_nids if n != pid and n in field.nodes]
                else:
                    break
        return merged_any
