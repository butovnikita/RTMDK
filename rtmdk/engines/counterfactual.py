"""rtmdk/engines/counterfactual.py"""
from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
from numpy.typing import NDArray

MemoryNode = Any


class ScenarioPlanner:
    """Counterfactual imagination and scenario planning."""

    def __init__(self, field: Any, max_scenarios: int = 5):
        self.field = field
        self.max_scenarios = max_scenarios

    def imagine_counterfactual(self, base_query: NDArray,
                               intervention: Dict[str, float]) -> List[Dict]:
        results = []
        for node_id, strength in list(intervention.items())[
                :self.max_scenarios]:
            if node_id not in self.field.nodes:
                continue
            node = self.field.nodes[node_id]
            hyp_phase = (node.phase + strength * np.pi) % (2 * np.pi)
            hyp_amp = min(1.0, node.amplitude * 1.2)
            hyp_sal = min(1.0, node.salience * 1.1)
            traj = self._simulate_trajectory(
                node, hyp_phase, hyp_amp, hyp_sal, steps=3)
            coherence = self._score_coherence(traj)
            results.append({
                "hypothetical": True,
                "node_id": node_id,
                "intervention_strength": strength,
                "trajectory": [t.tolist() if isinstance(t, np.ndarray) else t for t in traj],
                "confidence": coherence,
            })
        self.field.stats["scenarios_generated"] = self.field.stats.get(
            "scenarios_generated", 0) + len(results)
        if results:
            self.field.stats["avg_scenario_confidence"] = np.mean(
                [r["confidence"] for r in results])
        return results

    def _simulate_trajectory(
            self,
            node: MemoryNode,
            phase: float,
            amp: float,
            sal: float,
            steps: int = 3) -> List[NDArray]:
        traj = [node.latent_pos.copy()]
        current_pos = node.latent_pos.copy()
        for _ in range(steps):
            if self.field.ode_dynamics and len(self.field.nodes) > 1:
                all_positions = np.array(
                    [n.latent_pos for n in self.field.nodes.values()])
                state = all_positions.flatten()
                dynamics = self.field.ode_dynamics._dynamics(0, state)
                idx = list(self.field.nodes.keys()).index(node.id)
                update = dynamics[idx *
                                  self.field.cfg.latent_dim:(idx +
                                                             1) *
                                  self.field.cfg.latent_dim]
                current_pos = current_pos + update * 0.1
            else:
                current_pos = current_pos * 0.95 + \
                    np.random.randn(len(current_pos)).astype(np.float32) * 0.01
            traj.append(current_pos.copy())
        return traj

    def _score_coherence(self, trajectory: List[NDArray]) -> float:
        if len(trajectory) < 2:
            return 0.5
        dists = [np.linalg.norm(trajectory[i + 1] - trajectory[i])
                 for i in range(len(trajectory) - 1)]
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        coherence = np.exp(-mean_dist) * np.exp(-std_dist)
        return float(np.clip(coherence, 0.0, 1.0))
