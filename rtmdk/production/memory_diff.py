"""
rtmdk/production/memory_diff.py — Memory State Comparison.

Compares two memory states and reports differences.
"""

from typing import Dict, List, Any


class MemoryDiff:
    """Compares two memory states.
    
    Usage:
        diff = MemoryDiff(memory_before, memory_after)
        changes = diff.compute()
        # {"added": [...], "removed": [...], "modified": [...]}
    """
    
    def __init__(self, memory_before, memory_after):
        self.before = memory_before
        self.after = memory_after
    
    def compute(self) -> Dict[str, Any]:
        """Compute differences between two memory states."""
        before_nodes = set(self.before.field.nodes.keys())
        after_nodes = set(self.after.field.nodes.keys())
        
        added = after_nodes - before_nodes
        removed = before_nodes - after_nodes
        common = before_nodes & after_nodes
        
        modified = []
        for nid in common:
            b_node = self.before.field.nodes[nid]
            a_node = self.after.field.nodes[nid]
            
            changes = {}
            if b_node.salience != a_node.salience:
                changes["salience"] = {"before": b_node.salience, "after": a_node.salience}
            if b_node.amplitude != a_node.amplitude:
                changes["amplitude"] = {"before": b_node.amplitude, "after": a_node.amplitude}
            if b_node.content.get("text") != a_node.content.get("text"):
                changes["text"] = True
            
            if changes:
                modified.append({"node_id": nid, "changes": changes})
        
        return {
            "added": list(added),
            "removed": list(removed),
            "modified": modified,
            "summary": {
                "nodes_before": len(before_nodes),
                "nodes_after": len(after_nodes),
                "added_count": len(added),
                "removed_count": len(removed),
                "modified_count": len(modified),
            }
        }
