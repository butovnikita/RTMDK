"""CognitiveManager — self-supervision, TDA, state encoding, cognitive compression.

Extracted from RTMDKField to reduce monolithic field.py size.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField
    from rtmdk.nodes import MemoryNode
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class CognitiveManager:
    """Self-supervision, TDA monitoring, field state encoding, cognitive compression."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field

    def self_supervise(self) -> None:
        """Use local probe instead of np.zeros to prevent false decay of peripheral nodes."""
        f = self.field
        if not f.cfg.self_supervision:
            return
        f.stats["self_sup_checks"] += 1
        for nid in list(f.node_index):
            if nid not in f.nodes or not f.nodes[nid].lineage:
                continue
            node = f.nodes[nid]
            probe = node.latent_pos + f._rng.normal(0, 0.05, node.latent_pos.shape)
            results = f.query(probe, phase=node.phase, top_k=1)
            if results and results[0][0] == nid:
                node.self_sup_score = max(0.5, results[0][1])
            else:
                node.self_sup_score *= 0.9

    def check_tda(self) -> None:
        f = self.field
        if not f.cfg.tda_monitoring or not f.tda_monitor:
            return
        f.stats["tda_checks"] += 1
        r = f.tda_monitor.compute_persistence(f.nodes)
        f.stats["tda_H0"] = r["H0"]
        f.stats["tda_H1"] = r["H1"]
        if f.tda_monitor.get_trend() == "growing_contradictions":
            f.consolidate()

    def encode_field_state(self) -> NDArray:
        """Encode field state into a flat vector for predictive coding."""
        f = self.field
        if not f.nodes:
            return np.zeros(f.cfg.latent_dim * 4, dtype=np.float32)
        positions = np.array([n.latent_pos for n in f.nodes.values()])
        phases = np.array([n.phase for n in f.nodes.values()])
        amps = np.array([n.amplitude for n in f.nodes.values()])
        sals = np.array([n.salience for n in f.nodes.values()])
        mean_pos = np.mean(positions, axis=0)
        mean_phase = np.mean(phases)
        mean_amp = np.mean(amps)
        mean_sal = np.mean(sals)
        state = np.zeros(f.cfg.latent_dim * 4, dtype=np.float32)
        pos_dim = min(len(mean_pos), f.cfg.latent_dim)
        state[:pos_dim] = mean_pos[:pos_dim]
        state[f.cfg.latent_dim] = mean_phase
        state[f.cfg.latent_dim * 2] = mean_amp
        state[f.cfg.latent_dim * 3] = mean_sal
        return state

    def cognitive_compress(self, results: List[Tuple[str, float, MemoryNode]]) -> str:
        """Compress raw memory results into a structured cognitive dump for LLM."""
        f = self.field
        if not results:
            return "### COGNITIVE_CONTEXT\nNo relevant structures."

        high_res = [(nid, r, n) for nid, r, n in results if r > f.cfg.high_resonance_threshold]
        contradictions = [n for _, _, n in results if n.content.get("causal_flag") == "incompatible"]
        procedural = [n for _, _, n in results if getattr(n, 'tier', 'semantic') == "procedural"]

        lines = ["### COGNITIVE_CONTEXT"]
        if high_res:
            summaries = []
            for nid, r, n in high_res:
                text = n.content.get("text", "unknown")[:60]
                summaries.append(f"[{text}...](R:{r:.2f},S:{n.salience:.2f})")
            lines.append(f"• High resonance ({len(high_res)} nodes): " + " | ".join(summaries))
        if contradictions:
            texts = [n.content.get("text", "unknown")[:40] for n in contradictions[:3]]
            lines.append("[WARN] Conflicting nodes: " + " | ".join(texts))
        if procedural:
            lines.append("[TOOL] Procedural patterns available (how-to)")

        lineage_nodes = [(nid, n) for nid, r, n in results if n.lineage]
        if lineage_nodes:
            lines.append(f"[STATS] Consolidated memories: {len(lineage_nodes)} nodes with synthesis history")

        return "\n".join(lines)
