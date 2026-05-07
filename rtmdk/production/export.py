"""
rtmdk/production/export.py — Memory Export to PDF/Markdown/Text.

Exports RTMDK memory in readable formats.
"""

import time
from typing import Dict, Any
from pathlib import Path


class MemoryExporter:
    """Exports RTMDK memory to various formats.

    Usage:
        exporter = MemoryExporter(memory)

        # Export to markdown
        md = exporter.to_markdown()

        # Export to text
        txt = exporter.to_text()

        # Export to dict (for JSON)
        data = exporter.to_dict()
    """

    def __init__(self, memory):
        self.memory = memory

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary (for JSON serialization)."""
        nodes = {}
        for nid, node in self.memory.field.nodes.items():
            nodes[nid] = {
                "text": node.content.get("text", ""),
                "salience": node.salience,
                "amplitude": node.amplitude,
                "phase": node.phase,
                "tier": getattr(node, 'tier', 'unknown'),
                "created_at": node.created_at,
            }

        return {
            "exported_at": time.time(),
            "node_count": len(nodes),
            "nodes": nodes,
            "stats": dict(self.memory.field.stats),
        }

    def to_markdown(self, max_nodes: int = 100) -> str:
        """Export as Markdown document."""
        nodes = list(self.memory.field.nodes.values())
        nodes.sort(key=lambda n: n.salience, reverse=True)
        nodes = nodes[:max_nodes]

        lines = [
            "# RTMDK Memory Export",
            f"\nExported: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total nodes: {len(self.memory.field.nodes)}\n",
        ]

        # Group by tier
        tiers = {}
        for node in nodes:
            tier = getattr(node, 'tier', 'unknown')
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append(node)

        for tier, tier_nodes in tiers.items():
            lines.append(
                f"\n## {tier.title()} Memory ({len(tier_nodes)} nodes)\n")
            for node in tier_nodes:
                text = node.content.get("text", "")[:200]
                lines.append(f"- **[{node.salience:.2f}]** {text}")

        return '\n'.join(lines)

    def to_text(self, max_nodes: int = 100) -> str:
        """Export as plain text."""
        nodes = list(self.memory.field.nodes.values())
        nodes.sort(key=lambda n: n.salience, reverse=True)
        nodes = nodes[:max_nodes]

        lines = [
            "RTMDK Memory Export",
            f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Nodes: {len(self.memory.field.nodes)}\n",
        ]

        for node in nodes:
            text = node.content.get("text", "")[:200]
            lines.append(f"[{node.salience:.2f}] {text}")

        return '\n'.join(lines)

    def export_to_file(self, filepath: str, format: str = "markdown"):
        """Export memory to a file.

        Args:
            filepath: Output file path
            format: 'markdown', 'text', or 'json'
        """
        path = Path(filepath)

        if format == "markdown":
            content = self.to_markdown()
        elif format == "text":
            content = self.to_text()
        elif format == "json":
            import json
            content = json.dumps(self.to_dict(), indent=2, default=str)
        else:
            raise ValueError(f"Unknown format: {format}")

        path.write_text(content, encoding='utf-8')
        return str(path)
