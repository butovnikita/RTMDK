"""Kuramoto synchronization and federated sync for RTMDK."""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from rtmdk.nodes import FederatedNode

if TYPE_CHECKING:
    pass


class KuramotoSync:
    def __init__(self, coupling_strength: float = 0.5, dt: float = 0.01):
        self.coupling_strength = coupling_strength
        self.dt = dt
        self.phases: Dict[str, float] = {}
        self.natural_freqs: Dict[str, float] = {}
        self._order_history: deque = deque(maxlen=100)

    def add_oscillator(
            self,
            node_id: str,
            phase: float,
            natural_freq: float = 1.0):
        self.phases[node_id] = phase
        self.natural_freqs[node_id] = natural_freq

    def remove_oscillator(self, node_id: str):
        self.phases.pop(node_id, None)
        self.natural_freqs.pop(node_id, None)

    def step(self, n_steps: int = 1) -> Dict[str, float]:
        for _ in range(n_steps):
            new_phases = {}
            n = len(self.phases)
            if n < 2:
                continue
            K_over_N = self.coupling_strength / n
            for nid, phi in self.phases.items():
                omega = self.natural_freqs.get(nid, 1.0)
                coupling = 0.0
                for other_id, other_phi in self.phases.items():
                    if other_id != nid:
                        coupling += math.sin(other_phi - phi)
                new_phases[nid] = (
                    phi + self.dt * (omega + K_over_N * coupling)) % (2 * math.pi)
            self.phases.update(new_phases)
        self._order_history.append(self.compute_order_parameter())
        return self.phases

    def compute_order_parameter(self) -> float:
        if not self.phases:
            return 0.0
        n = len(self.phases)
        sum_exp = sum(complex(math.cos(p), math.sin(p))
                      for p in self.phases.values())
        return abs(sum_exp) / n

    def sync_to_target(
            self, target_phases: Dict[str, float], n_steps: int = 10) -> Dict[str, float]:
        for nid, target_phi in target_phases.items():
            if nid in self.phases:
                diff = target_phi - self.phases[nid]
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                self.phases[nid] = (
                    self.phases[nid] + self.coupling_strength * diff) % (2 * math.pi)
        for _ in range(n_steps):
            self.step()
        return self.phases

    def get_state(self) -> Dict:
        return {
            "phases": dict(
                self.phases),
            "natural_freqs": dict(
                self.natural_freqs),
            "order_parameter": self.compute_order_parameter(),
            "coupling_strength": self.coupling_strength,
        }

    def load_state(self, state: Dict):
        self.phases = state.get("phases", {})
        self.natural_freqs = state.get("natural_freqs", {})
        self.coupling_strength = state.get(
            "coupling_strength", self.coupling_strength)


class FederatedRTMDK:
    def __init__(self, node_id: str = "local", sync_lr: float = 0.01,
                 sync_freq: int = 100, min_resonance: float = 0.2,
                 coupling_strength: float = 0.5):
        self.node_id = node_id
        self.sync_lr = sync_lr
        self.sync_freq = sync_freq
        self.min_resonance = min_resonance
        self.kuramoto = KuramotoSync(coupling_strength=coupling_strength)
        self.peers: Dict[str, FederatedNode] = {}
        self._sync_history: List[Dict] = []
        self._total_syncs = 0
        self._step_counter = 0

    def register_peer(self, peer: FederatedNode):
        self.peers[peer.node_id] = peer
        self.kuramoto.add_oscillator(
            peer.node_id, peer.phase, peer.natural_freq)

    def unregister_peer(self, peer_id: str):
        self.peers.pop(peer_id, None)
        self.kuramoto.remove_oscillator(peer_id)

    def sync_with_peers(self, local_phases: Dict[str, float],
                        local_params: Dict[str, float]) -> Dict[str, Any]:
        self._step_counter += 1
        if self._step_counter % self.sync_freq != 0:
            return {"synced": False, "reason": "not_sync_step"}
        if not self.peers:
            return {"synced": False, "reason": "no_peers"}
        sync_results = []
        for peer_id, peer in self.peers.items():
            if not peer.is_active:
                continue
            resonance = self._compute_param_resonance(
                local_params, peer.params)
            if resonance < self.min_resonance:
                continue
            self.kuramoto.sync_to_target(local_phases, n_steps=5)
            blended_params = self._blend_params(
                local_params, peer.params, self.sync_lr)
            peer.params = blended_params
            peer.sync_count += 1
            peer.last_sync_time = time.time()
            sync_results.append({
                "peer_id": peer_id, "resonance": resonance,
                "params_updated": list(blended_params.keys()),
            })
        self._total_syncs += 1
        self._sync_history.append({
            "time": time.time(), "peers_synced": len(sync_results),
            "order_parameter": self.kuramoto.compute_order_parameter(),
        })
        return {
            "synced": True, "results": sync_results,
            "order_parameter": self.kuramoto.compute_order_parameter(),
            "total_syncs": self._total_syncs,
        }

    def _compute_param_resonance(self, params_a: Dict[str, float],
                                 params_b: Dict[str, float]) -> float:
        common_keys = set(params_a.keys()) & set(params_b.keys())
        if not common_keys:
            return 0.0
        diffs = []
        for key in common_keys:
            a, b = params_a[key], params_b[key]
            denom = max(abs(a) + abs(b), 1e-8)
            diffs.append(1.0 - abs(a - b) / denom)
        return float(np.mean(diffs))

    def _blend_params(self,
                      params_a: Dict[str,
                                     float],
                      params_b: Dict[str,
                                     float],
                      lr: float) -> Dict[str,
                                         float]:
        blended = {}
        all_keys = set(params_a.keys()) | set(params_b.keys())
        for key in all_keys:
            a = params_a.get(key, 0.0)
            b = params_b.get(key, 0.0)
            blended[key] = (1 - lr) * a + lr * b
        return blended

    def get_aggregated_params(self) -> Dict[str, float]:
        all_params: Dict[str, List[float]] = defaultdict(list)
        for peer in self.peers.values():
            if peer.is_active:
                for k, v in peer.params.items():
                    all_params[k].append(v)
        return {k: float(np.mean(v)) for k, v in all_params.items() if v}

    def get_sync_status(self) -> Dict:
        return {
            "node_id": self.node_id, "n_peers": len(self.peers),
            "active_peers": sum(1 for p in self.peers.values() if p.is_active),
            "order_parameter": self.kuramoto.compute_order_parameter(),
            "total_syncs": self._total_syncs,
            "sync_history": self._sync_history[-10:],
            "kuramoto_state": self.kuramoto.get_state(),
        }

    def export_state(self) -> Dict:
        return {
            "node_id": self.node_id,
            "peers": {
                pid: p.to_dict() for pid,
                p in self.peers.items()},
            "kuramoto": self.kuramoto.get_state(),
            "sync_history": self._sync_history,
            "total_syncs": self._total_syncs,
        }

    def import_state(self, state: Dict):
        self.node_id = state.get("node_id", self.node_id)
        self._total_syncs = state.get("total_syncs", 0)
        self._sync_history = state.get("sync_history", [])
        for pid, pdata in state.get("peers", {}).items():
            peer = FederatedNode.from_dict(pdata)
            self.peers[pid] = peer
            self.kuramoto.add_oscillator(pid, peer.phase, peer.natural_freq)
        if "kuramoto" in state:
            self.kuramoto.load_state(state["kuramoto"])
