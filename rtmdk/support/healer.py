"""Topology healer for RTMDK."""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from rtmdk.memory.config import FieldHealth
    from rtmdk.nodes import MemoryNode


class TopologyHealer:
    def __init__(
            self,
            dead_zone_threshold: float = 0.15,
            hyperconvergence_threshold: float = 0.05,
            fragmentation_threshold: float = 0.6,
            healing_strength: float = 0.1,
            max_healing_nodes: int = 5):
        self.dead_zone_threshold = dead_zone_threshold
        self.hyperconvergence_threshold = hyperconvergence_threshold
        self.fragmentation_threshold = fragmentation_threshold
        self.healing_strength = healing_strength
        self.max_healing_nodes = max_healing_nodes
        self._health_history: deque = deque(maxlen=100)

    def detect_dead_zones(self, nodes: Dict[str, "MemoryNode"]) -> List[str]:
        if len(nodes) < 3:
            return []
        positions = np.array([n.latent_pos for n in nodes.values()])
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists, axis=1)
        threshold = np.median(min_dists) * (1.0 + self.dead_zone_threshold * 5)
        return [nid for i, nid in enumerate(nodes) if min_dists[i] > threshold]

    def detect_hyperconvergence(self, nodes: Dict[str, "MemoryNode"]) -> bool:
        if len(nodes) < 3:
            return False
        positions = np.array([n.latent_pos for n in nodes.values()])
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        return bool(np.mean(dists[dists < np.inf]) < self.hyperconvergence_threshold)

    def detect_fragmentation(
            self, nodes: Dict[str, "MemoryNode"], radius: float = 2.0) -> float:
        if len(nodes) < 2:
            return 0.0
        positions = np.array([n.latent_pos for n in nodes.values()])
        dists = cdist(positions, positions)
        np.fill_diagonal(dists, np.inf)
        isolated: int = int(np.sum(np.all(dists > radius, axis=1)))
        return float(isolated / len(nodes))

    def compute_field_health(
            self, nodes: Dict[str, "MemoryNode"]) -> Tuple["FieldHealth", Dict]:
        from rtmdk.memory.config import FieldHealth
        diagnostics: Dict[str, Any] = {}
        dead = self.detect_dead_zones(nodes)
        diagnostics["dead_zones"] = len(dead)
        diagnostics["dead_zone_nodes"] = dead
        hyperconv = self.detect_hyperconvergence(nodes)
        diagnostics["hyperconvergence"] = hyperconv
        frag = self.detect_fragmentation(nodes)
        diagnostics["fragmentation"] = frag
        if len(nodes) >= 3:
            positions = np.array([n.latent_pos for n in nodes.values()])
            dists = cdist(positions, positions)
            np.fill_diagonal(dists, np.inf)
            valid = dists[dists < np.inf]
            diagnostics["avg_pairwise_dist"] = float(np.mean(valid))
            diagnostics["std_pairwise_dist"] = float(np.std(valid))
            diagnostics["density_cv"] = float(
                np.std(valid) / float(max(np.mean(valid), 1e-8)))  # type: ignore[arg-type]
        else:
            diagnostics["avg_pairwise_dist"] = 0.0
            diagnostics["density_cv"] = 0.0
        if hyperconv or frag > 0.8:
            health = FieldHealth.CRITICAL
        elif len(dead) > len(nodes) * 0.3 or frag > self.fragmentation_threshold:
            health = FieldHealth.DEGRADED
        else:
            health = FieldHealth.STABLE
        self._health_history.append(health.value)
        diagnostics["health"] = health.value
        diagnostics["stable_fraction"] = (sum(
            1 for h in self._health_history if h == "stable") / max(len(self._health_history), 1))
        return health, diagnostics

    def heal_dead_zones(self,
                        nodes: Dict[str,
                                    "MemoryNode"],
                        dead_ids: List[str]) -> List[Dict]:
        healed: List[Dict[str, Any]] = []
        alive_ids = [nid for nid in nodes if nid not in dead_ids]
        if not alive_ids or not dead_ids:
            return healed
        alive_positions = np.array(
            [nodes[nid].latent_pos for nid in alive_ids])
        for dead_id in dead_ids[:self.max_healing_nodes]:
            dead_node = nodes[dead_id]
            dists = np.linalg.norm(
                alive_positions - dead_node.latent_pos, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_id = alive_ids[nearest_idx]
            old_pos = dead_node.latent_pos.copy()
            dead_node.latent_pos = (
                (1.0 -
                 self.healing_strength) *
                old_pos +
                self.healing_strength *
                nodes[nearest_id].latent_pos).astype(
                np.float32)
            dead_node.is_healing = True
            dead_node.healing_origin = nearest_id
            dead_node.salience = max(dead_node.salience, 0.1)
            dead_node.amplitude = max(dead_node.amplitude, 0.1)
            healed.append({"node_id": dead_id,
                           "from": old_pos.tolist(),
                           "to": dead_node.latent_pos.tolist(),
                           "type": "dead_zone"})
        return healed

    def heal_hyperconvergence(
            self, nodes: Dict[str, "MemoryNode"]) -> List[Dict]:
        healed: List[Dict[str, Any]] = []
        if len(nodes) < 3:
            return healed
        positions = np.array([n.latent_pos for n in nodes.values()])
        centroid = np.mean(positions, axis=0)
        for nid in list(nodes.keys())[:self.max_healing_nodes]:
            node = nodes[nid]
            direction = node.latent_pos - centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = np.random.randn(len(centroid)).astype(np.float32)
                norm_val: float = 1.0
            else:
                norm_val = float(norm)
            direction = direction / norm_val
            old_pos = node.latent_pos.copy()
            node.latent_pos = (
                old_pos +
                self.healing_strength *
                direction).astype(
                np.float32)
            node.is_healing = True
            node.healing_origin = "hyperconvergence"
            healed.append({"node_id": nid,
                           "from": old_pos.tolist(),
                           "to": node.latent_pos.tolist(),
                           "type": "hyperconvergence"})
        return healed

    def heal_fragmentation(self,
                           nodes: Dict[str,
                                       "MemoryNode"],
                           isolated_ids: List[str]) -> List[Dict]:
        healed: List[Dict[str, Any]] = []
        non_isolated = [nid for nid in nodes if nid not in isolated_ids]
        if not non_isolated or not isolated_ids:
            return healed
        non_iso_positions = np.array(
            [nodes[nid].latent_pos for nid in non_isolated])
        centroid = np.mean(non_iso_positions, axis=0)
        for iso_id in isolated_ids[:self.max_healing_nodes]:
            node = nodes[iso_id]
            old_pos = node.latent_pos.copy()
            node.latent_pos = (
                (1.0 -
                 self.healing_strength) *
                old_pos +
                self.healing_strength *
                centroid).astype(
                np.float32)
            node.is_healing = True
            node.healing_origin = "fragmentation"
            healed.append({"node_id": iso_id,
                           "from": old_pos.tolist(),
                           "to": node.latent_pos.tolist(),
                           "type": "fragmentation"})
        return healed

    def get_state(self) -> Dict:
        return {"health_history": list(self._health_history), "stable_fraction": sum(
            1 for h in self._health_history if h == "stable") / max(len(self._health_history), 1)}

    def load_state(self, state: Dict):
        self._health_history = deque(
            state.get("health_history", []), maxlen=100)
