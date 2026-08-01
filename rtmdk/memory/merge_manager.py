"""rtmdk/memory/merge_manager.py

Learned and heuristic latent merge logic for RTMDKField consolidation.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField
    from rtmdk.nodes import MemoryNode


class MergeManager:
    """Encapsulates node latent merge and consolidator training."""

    def __init__(self, field: RTMDKField):
        self._field = field

    def merge_latents(self, node: MemoryNode, partner: MemoryNode):
        """Merge two node latent positions using learned or heuristic method."""
        field = self._field
        if field.learned_consolidator is not None and field.learned_consolidator._trained:
            # Undo quantization for learned merge (needs float32 latent)
            latent_a = field._quant.dequantize(node.latent_pos, node.latent_scale, node.latent_zero_point)
            latent_b = field._quant.dequantize(partner.latent_pos, partner.latent_scale, partner.latent_zero_point)
            merged = field.learned_consolidator.predict(
                latent_a,
                latent_b,
                node.phase,
                partner.phase,
                node.amplitude,
                partner.amplitude,
                node.salience,
                partner.salience,
            )
            # Re-quantize
            merged_q, scale, zp = field._quant.quantize_with_meta(merged)
            node.latent_pos = merged_q
            node.latent_scale = scale
            node.latent_zero_point = zp
        else:
            # Heuristic average (preserves existing behaviour)
            node.latent_pos = 0.5 * (node.latent_pos + partner.latent_pos)

    def train_learned_consolidator(self):
        """Collect synthetic merge examples and train the consolidator MLP."""
        field = self._field
        if field.learned_consolidator is None:
            return
        n = len(field.nodes)
        if n < 4:
            return
        # Sample random node pairs as synthetic merge examples
        rng = np.random.default_rng(42)
        node_list = list(field.nodes.values())
        for _ in range(min(50, n * 2)):
            a, b = rng.choice(node_list, size=2, replace=False)
            # Dequantize latents for training
            la = field._quant.dequantize(a.latent_pos, a.latent_scale, a.latent_zero_point)
            lb = field._quant.dequantize(b.latent_pos, b.latent_scale, b.latent_zero_point)
            # Queries = the parent latents themselves (proxy)
            field.learned_consolidator.add_example(
                la,
                lb,
                queries=[la, lb],
                phase_a=a.phase,
                phase_b=b.phase,
                amp_a=a.amplitude,
                amp_b=b.amplitude,
                sal_a=a.salience,
                sal_b=b.salience,
            )
        field.learned_consolidator.train(epochs=10, lr=0.005)
