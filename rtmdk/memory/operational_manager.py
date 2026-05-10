"""rtmdk/memory/operational_manager.py

Operational methods for RTMDKField: calibration, health checks,
interventions, counterfactual queries, rollbacks.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from rtmdk.memory.config import FieldHealth

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField
    from rtmdk.nodes import CounterfactualResult


class OperationalManager:
    """Encapsulates operational/health/intervention methods for RTMDKField."""

    def __init__(self, field: RTMDKField):
        self._field = field

    def calibrate(
            self,
            query_embedding: NDArray,
            node_id: str,
            is_relevant: bool) -> None:
        """Add a labeled query-result pair to the conformal calibration set."""
        field = self._field
        if not field.cfg.conformal_prediction or field.conformal_calibrator is None:
            return
        if not is_relevant or node_id not in field.nodes:
            return
        node = field.nodes[node_id]
        query_latent = field._projection_mgr.project(query_embedding)
        score = field._query_mgr._resonance_response(query_latent, node.phase, node)
        field.conformal_calibrator.add_sample(score)

    def imagine_counterfactual(self, base_query: NDArray,
                               intervention: Dict[str, float]) -> List[Dict]:
        """Generate hypothetical trajectories via do-interventions."""
        field = self._field
        if not field.scenario_planner:
            return []
        return field.scenario_planner.imagine_counterfactual(base_query, intervention)

    def rollback_consolidation(self, n_steps: int = 1) -> bool:
        """Rollback the last N consolidation operations."""
        field = self._field
        if not field._rollback_history or n_steps > len(field._rollback_history):
            return False
        snapshot = field._rollback_history[-n_steps]
        for nid, state in snapshot["pre_state"].items():
            if nid in field.nodes:
                node = field.nodes[nid]
                node.latent_pos = state["latent_pos"].copy()
                node.phase = state["phase"]
                node.amplitude = state["amplitude"]
                node.salience = state["salience"]
                node.tension = state.get("tension", 0.0)
                node.soft_gate = state.get("soft_gate", 1.0)
                if "content" in state:
                    node.content = dict(state["content"])
                if "lineage" in state:
                    node.lineage = list(state["lineage"])
                if "causal_strength" in state:
                    node.causal_strength = dict(state["causal_strength"])
                if "causal_parents" in state:
                    node.causal_parents = list(state["causal_parents"])
                node.pre_consolidation_pos = None
        field._rollback_history = field._rollback_history[:-n_steps]
        field._tension_cache.clear()
        return True

    def do_intervention(self, node_id: str, new_embedding: NDArray):
        """Apply a do-intervention to a node's latent position."""
        field = self._field
        if node_id not in field.nodes:
            return
        new_pos = field._projection_mgr.project(new_embedding)
        if field.causal_engine:
            field.causal_engine.do_intervention(node_id, new_pos)
        field.nodes[node_id].latent_pos = new_pos

    def clear_interventions(self):
        """Clear all active do-interventions."""
        field = self._field
        if field.causal_engine:
            field.causal_engine.clear_interventions()

    def get_field_health(self) -> Dict:
        """Compute and return field health diagnostics."""
        field = self._field
        if field.healer:
            health, diagnostics = field.healer.compute_field_health(field.nodes)
            diagnostics["kurtosis"] = field.stats.get("meta_kurtosis", 3.0)
            return diagnostics
        return {"health": "unknown", "kurtosis": 3.0}

    def self_heal(self) -> List[Dict]:
        """Run self-healing on the field (dead zones, hyperconvergence, fragmentation)."""
        field = self._field
        if not field.healer or len(field.nodes) < 3:
            return []
        health, diagnostics = field.healer.compute_field_health(field.nodes)
        field.stats["field_health"] = health.value
        healed = []
        if health == FieldHealth.STABLE:
            for nid in field.node_index:
                field.nodes[nid].is_healing = False
                field.nodes[nid].healing_origin = None
            return []
        field.stats["field_health"] = FieldHealth.HEALING.value
        if diagnostics.get("dead_zones", 0) > 0:
            healed.extend(
                field.healer.heal_dead_zones(
                    field.nodes,
                    diagnostics["dead_zone_nodes"]))
        if diagnostics.get("hyperconvergence", False):
            healed.extend(field.healer.heal_hyperconvergence(field.nodes))
        if diagnostics.get("fragmentation", 0) > field.cfg.fragmentation_threshold:
            if len(field.nodes) >= 2:
                positions = np.array(
                    [n.latent_pos for n in field.nodes.values()])
                tree = cKDTree(positions)
                neighbors = tree.query_ball_point(positions, 2.0)
                isolated = [field.node_index[i] for i in range(
                    len(field.node_index)) if len(neighbors[i]) <= 1]
                if isolated:
                    healed.extend(
                        field.healer.heal_fragmentation(
                            field.nodes, isolated))
        if healed:
            field.stats["healing_events"] += len(healed)
            field.stats["healing_history"].extend(healed)
            if len(field.stats["healing_history"]) > 1000:
                field.stats["healing_history"] = field.stats["healing_history"][-500:]
        return healed

    def counterfactual_query(self,
                             intervention: Dict[str, Any],
                             query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        """Run a counterfactual query against the causal engine."""
        field = self._field
        if not field.causal_engine:
            return CounterfactualResult(
                query=str(intervention),
                intervention=intervention,
                predicted_outcomes=[],
                confidence=0.0,
                reasoning_path=["Causal engine not enabled"],
                assumptions=[])
        field.stats["counterfactual_queries"] += 1
        return field.causal_engine.counterfactual_query(
            intervention, query_nodes, evidence, field.cfg.counterfactual_max_depth)

    def get_causal_summary(self) -> Dict:
        """Return a summary of the causal graph state."""
        field = self._field
        if not field.causal_engine:
            return {"enabled": False}
        return {
            "enabled": True,
            "causal_edges": len(field.causal_engine.causal_effects),
            "contradictions": len([
                c for c in field.causal_engine.contradictions.values()
                if not c.resolved]),
            "nodes_with_effects": len(set(k[0] for k in field.causal_engine.causal_effects)),
            "nodes_affected": len(set(k[1] for k in field.causal_engine.causal_effects)),
            "top_effects": sorted(
                [(f"{k[0]}->{k[1]}", v.strength)
                 for k, v in field.causal_engine.causal_effects.items()],
                key=lambda x: x[1], reverse=True)[:10],
        }

    def compress_field(self) -> None:
        """Compress node latent positions via incremental SVD."""
        field = self._field
        if not field.low_rank_compressor or len(field.nodes) < 10:
            return
        positions = np.array([n.latent_pos for n in field.nodes.values()])
        compressed, reconstructed = field.low_rank_compressor.compress(positions)
        ratio = field.low_rank_compressor.get_compression_ratio(positions.shape)
        field.stats["compression_ratio"] = ratio
        field.stats["compression_updates"] = field.low_rank_compressor._update_count
        for i, nid in enumerate(field.node_index):
            if i < len(reconstructed) and nid in field.nodes:
                field.nodes[nid].latent_pos = reconstructed[i].astype(np.float32)
