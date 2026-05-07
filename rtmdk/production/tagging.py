"""
rtmdk/production/tagging.py — Memory Tagging System.

Allows adding custom tags to memory nodes for organization.
"""

from typing import Any, Dict, List, Set
from collections import defaultdict


class TaggingSystem:
    """Adds custom tags to memory nodes.

    Usage:
        tags = TaggingSystem(memory)

        # Add tags
        tags.add_tag("node_123", "important")
        tags.add_tags("node_123", ["coffee", "morning"])

        # Query by tag
        node_ids = tags.get_nodes_by_tag("coffee")

        # List all tags
        all_tags = tags.list_tags()
    """

    def __init__(self, memory: Any):
        self.memory = memory
        self._node_tags: Dict[str, Set[str]] = defaultdict(set)
        self._tag_nodes: Dict[str, Set[str]] = defaultdict(set)

    def add_tag(self, node_id: str, tag: str):
        """Add a tag to a node."""
        self._node_tags[node_id].add(tag)
        self._tag_nodes[tag].add(node_id)

    def add_tags(self, node_id: str, tags: List[str]):
        """Add multiple tags to a node."""
        for tag in tags:
            self.add_tag(node_id, tag)

    def remove_tag(self, node_id: str, tag: str):
        """Remove a tag from a node."""
        self._node_tags[node_id].discard(tag)
        self._tag_nodes[tag].discard(node_id)

    def get_nodes_by_tag(self, tag: str) -> List[str]:
        """Get all nodes with a specific tag."""
        return list(self._tag_nodes.get(tag, set()))

    def get_tags_for_node(self, node_id: str) -> List[str]:
        """Get all tags for a node."""
        return list(self._node_tags.get(node_id, set()))

    def list_tags(self) -> Dict[str, int]:
        """List all tags with node counts."""
        return {tag: len(nodes) for tag, nodes in self._tag_nodes.items()}

    def export_tags(self) -> Dict[str, List[str]]:
        """Export all tags."""
        return {nid: list(tags) for nid, tags in self._node_tags.items()}

    def import_tags(self, data: Dict[str, List[str]]):
        """Import tags from exported data."""
        for node_id, tags in data.items():
            for tag in tags:
                self.add_tag(node_id, tag)
