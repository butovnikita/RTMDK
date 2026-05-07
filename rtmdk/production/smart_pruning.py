"""
rtmdk/production/smart_pruning.py — TTL-based Memory Cleanup.

Removes old and irrelevant nodes to keep memory efficient.
Features:
- TTL-based: nodes with salience < threshold AND age > N days
- Runs in background via OfflineDreamer
- Configurable thresholds per memory tier
- Export-before-pruning (safety net)
- Statistics: nodes pruned, RAM saved
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path


class SmartPruner:
    """Automatically prunes old and irrelevant memory nodes.

    Usage:
        pruner = SmartPruner(memory, max_age_days=90, min_salience=0.05)

        # Run pruning manually:
        result = pruner.prune()

        # Or integrate with OfflineDreamer for background pruning:
        dreamer = OfflineDreamer(field=memory.field, ...)
        dreamer.on_step()  # Includes pruning
    """

    def __init__(
        self,
        memory,  # RTMDKMemory instance
        max_age_days: int = 90,
        min_salience: float = 0.05,
        min_amplitude: float = 0.05,
        dry_run: bool = False,
        export_before_prune: bool = True,
        export_dir: str = "~/.rtmdk/pruning_backups",
        tier_overrides: Optional[Dict[str, Dict]] = None,
        on_prune: Optional[Callable] = None,
    ):
        self.memory = memory
        self.max_age_seconds = max_age_days * 86400
        self.min_salience = min_salience
        self.min_amplitude = min_amplitude
        self.dry_run = dry_run
        self.export_before_prune = export_before_prune
        self.export_dir = Path(export_dir).expanduser()
        self.on_prune = on_prune

        # Per-tier overrides
        self.tier_overrides = tier_overrides or {
            "episodic": {"max_age_days": 30, "min_salience": 0.1},
            "semantic": {"max_age_days": 365, "min_salience": 0.02},
            "procedural": {"max_age_days": 730, "min_salience": 0.01},
        }

        self._stats: Dict[str, Any] = {
            "total_prunes": 0,
            "total_nodes_pruned": 0,
            "last_prune": None,
            "last_pruned_nodes": 0,
        }

    def prune(self) -> Dict[str, Any]:
        """Run pruning. Returns dict with pruning stats.

        Returns:
            {
                "nodes_before": int,
                "nodes_after": int,
                "nodes_pruned": int,
                "ram_saved_mb": float,
                "dry_run": bool,
                "exported_to": str or None,
            }
        """
        nodes_before = len(self.memory.field.nodes)
        nodes_to_prune = []

        for nid, node in self.memory.field.nodes.items():
            if self._should_prune(node, nid):
                nodes_to_prune.append(nid)

        if not nodes_to_prune:
            return {
                "nodes_before": nodes_before,
                "nodes_after": nodes_before,
                "nodes_pruned": 0,
                "ram_saved_mb": 0.0,
                "dry_run": self.dry_run,
                "exported_to": None,
            }

        # Export before pruning (safety net)
        exported_to = None
        if self.export_before_prune and not self.dry_run:
            exported_to = self._export_nodes(nodes_to_prune)

        # Actually prune (or just report if dry_run)
        if not self.dry_run:
            for nid in nodes_to_prune:
                if nid in self.memory.field.nodes:
                    del self.memory.field.nodes[nid]
                if nid in self.memory.field.node_index:
                    self.memory.field.node_index.remove(nid)

        # Estimate RAM saved (~2KB per node)
        ram_saved = len(nodes_to_prune) * 2 / 1024  # MB

        # Update stats
        self._stats["total_prunes"] += 1
        self._stats["total_nodes_pruned"] += len(nodes_to_prune)
        self._stats["last_prune"] = time.time()
        self._stats["last_pruned_nodes"] = len(nodes_to_prune)

        result = {
            "nodes_before": nodes_before,
            "nodes_after": nodes_before - len(nodes_to_prune),
            "nodes_pruned": len(nodes_to_prune),
            "ram_saved_mb": round(ram_saved, 2),
            "dry_run": self.dry_run,
            "exported_to": exported_to,
        }

        # Callback
        if self.on_prune and not self.dry_run:
            self.on_prune(result)

        return result

    def _should_prune(self, node, node_id: str) -> bool:
        """Check if a node should be pruned."""
        # Check age
        age = time.time() - node.created_at
        tier = getattr(node, 'tier', 'semantic')

        # Get tier-specific thresholds
        tier_config = self.tier_overrides.get(tier, {})
        max_age = tier_config.get("max_age_days", 90) * 86400
        min_sal = tier_config.get("min_salience", self.min_salience)

        # Node must be BOTH old AND unimportant
        if age < max_age:
            return False
        if node.salience > min_sal:
            return False
        if node.amplitude > self.min_amplitude:
            return False

        return True

    def _export_nodes(self, node_ids: List[str]) -> Optional[str]:
        """Export nodes about to be pruned to a backup file."""
        if not node_ids:
            return None

        self.export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = self.export_dir / f"prune_backup_{timestamp}.json"

        data = []
        for nid in node_ids:
            node = self.memory.field.nodes.get(nid)
            if node:
                data.append({
                    "id": nid,
                    "text": node.content.get("text", ""),
                    "salience": node.salience,
                    "amplitude": node.amplitude,
                    "tier": getattr(node, 'tier', 'unknown'),
                    "created_at": node.created_at,
                })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def get_stats(self) -> Dict[str, Any]:
        """Get pruning statistics."""
        return {
            **self._stats,
            "config": {
                "max_age_days": self.max_age_seconds / 86400,
                "min_salience": self.min_salience,
                "min_amplitude": self.min_amplitude,
                "dry_run": self.dry_run,
            },
            "tier_overrides": self.tier_overrides,
        }
