"""CrystallizationManager — detect recurring patterns and crystallize.

Extracted from RTMDKField to reduce monolithic field.py size.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField


class CrystallizationManager:
    """Detects recurring episodic patterns and crystallizes into semantic nodes."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field

    def crystallize_recurring(self, window: int = 100, similarity_thresh: float = 0.75) -> None:
        """Detect recurring episodic patterns and crystallize into semantic nodes."""
        f = self.field
        recent_ids = f.node_index[-window:]
        recent = [
            f.nodes[nid]
            for nid in recent_ids
            if nid in f.nodes
            and getattr(f.nodes[nid], "tier", "semantic") == "episodic"
            and nid not in f._crystallized_nodes
        ]
        if len(recent) < 5:
            return

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return

        pos = np.array([n.latent_pos for n in recent])
        labels = DBSCAN(eps=0.4, min_samples=f.cfg.crystallization_min_cluster).fit_predict(pos)

        crystallized_count = 0
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            members = [recent[i] for i, l in enumerate(labels) if l == cluster_id]
            if len(members) >= f.cfg.crystallization_min_cluster:
                new_pos = np.mean([m.latent_pos for m in members], axis=0).astype(np.float32)
                phases = np.array([m.phase for m in members])
                new_phase = float(np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))) % (2 * np.pi)
                combined_text = " ".join([m.content.get("text", "")[:30] for m in members[:3]])
                new_content = {
                    "text": f"Crystallized: {combined_text}...",
                    "tier": "semantic",
                    "crystallized_from": [m.id for m in members],
                    "crystallized_at": time.time(),
                }
                new_id = f.add_node(new_pos, new_content, phase=float(new_phase % (2 * np.pi)), skip_projection=True)
                f.nodes[new_id].tier = "semantic"
                for m in members:
                    m.content["archived"] = True
                    f._crystallized_nodes.add(m.id)
                crystallized_count += 1

        if crystallized_count > 0:
            f.stats["crystallizations"] += crystallized_count
            f.stats["crystallized_clusters"] += crystallized_count
