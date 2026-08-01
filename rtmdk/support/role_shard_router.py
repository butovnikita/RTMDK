"""rtmdk/support/role_shard_router.py — Role-based field sharding with intra-shard Kuramoto.

Splits the memory field into role-based shards. Within each shard:
- Full Kuramoto synchronization (attractors of the same role reinforce each other)
- Full consolidation and resonance

Between shards:
- Exchange only when cross_role_resonance > threshold
- Queries route to relevant shard + fallback to others

For single-user local chat: one `default` shard, ~zero overhead.
For multi-tenant: shards by user/role with full isolation by default.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import numpy as np

# Role keywords for auto-detection
ROLE_KEYWORDS: Dict[str, List[str]] = {
    "coding": [
        "def ",
        "class ",
        "function",
        "import ",
        "from ",
        "return ",
        "code",
        "function",
        "variable",
        "algorithm",
        "debug",
        "compile",
        "python",
        "javascript",
        "typescript",
        "rust",
        "go ",
        "java",
        "api",
        "endpoint",
        "middleware",
        "docker",
        "kubernetes",
        "git",
        "async",
        "await",
        "lambda",
        "yield",
        "try",
        "except",
    ],
    "work": [
        "meeting",
        "deadline",
        "project",
        "task",
        "report",
        "presentation",
        "client",
        "customer",
        "team",
        "manager",
        "office",
        "email",
        "собрание",
        "дедлайн",
        "проект",
        "задача",
        "отчёт",
        "команда",
    ],
    "personal": [
        "family",
        "friend",
        "hobby",
        "travel",
        "food",
        "health",
        "fitness",
        "movie",
        "book",
        "music",
        "game",
        "weekend",
        "vacation",
        "семья",
        "друг",
        "хобби",
        "путешеств",
        "еда",
        "здоровь",
        "фильм",
        "книг",
        "музык",
        "игр",
        "выходн",
        "отпуск",
    ],
    "research": [
        "research",
        "study",
        "paper",
        "experiment",
        "hypothesis",
        "theory",
        "analysis",
        "data",
        "model",
        "train",
        "learn",
        "neural",
        "исследован",
        "эксперимент",
        "гипотез",
        "теор",
        "анализ",
        "данны",
        "модель",
        "обучен",
        "нейрон",
    ],
    "learning": [
        "learn",
        "study",
        "course",
        "tutorial",
        "how to",
        "explain",
        "understand",
        "concept",
        "practice",
        "exercise",
        "учить",
        "курс",
        "объясн",
        "понять",
        "концепц",
        "упражнен",
    ],
}

# Default role when auto-detection fails
DEFAULT_ROLE = "default"


@dataclass
class RoleShard:
    """A single role-based shard."""

    role: str
    node_ids: Set[str] = field(default_factory=set)
    kuramoto_phases: Dict[str, float] = field(default_factory=dict)
    kuramoto_coupling: float = 0.3
    last_sync_time: float = field(default_factory=time.time)
    n_queries: int = 0
    n_consolidations: int = 0
    n_cross_shard_exchanges: int = 0

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "node_ids": list(self.node_ids),
            "kuramoto_phases": dict(self.kuramoto_phases),
            "kuramoto_coupling": self.kuramoto_coupling,
            "last_sync_time": self.last_sync_time,
            "n_queries": self.n_queries,
            "n_consolidations": self.n_consolidations,
            "n_cross_shard_exchanges": self.n_cross_shard_exchanges,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RoleShard":
        shard = cls(
            role=data["role"],
            node_ids=set(data.get("node_ids", [])),
            kuramoto_phases=data.get("kuramoto_phases", {}),
            kuramoto_coupling=data.get("kuramoto_coupling", 0.3),
            last_sync_time=data.get("last_sync_time", time.time()),
            n_queries=data.get("n_queries", 0),
            n_consolidations=data.get("n_consolidations", 0),
            n_cross_shard_exchanges=data.get("n_cross_shard_exchanges", 0),
        )
        return shard


class RoleDetector:
    """Auto-detects role from text content."""

    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None):
        self.keywords = keywords or ROLE_KEYWORDS
        # Compile patterns for performance
        self._patterns: Dict[str, List[re.Pattern]] = {}
        for role, words in self.keywords.items():
            patterns = []
            for word in words:
                patterns.append(re.compile(re.escape(word), re.IGNORECASE))
            self._patterns[role] = patterns

    def detect(self, text: str) -> str:
        """Detect role from text. Returns role with most keyword matches."""
        if not text or len(text) < 3:
            return DEFAULT_ROLE
        text_lower = text.lower()
        scores: Dict[str, int] = defaultdict(int)
        for role, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    scores[role] += 1
        if not scores:
            return DEFAULT_ROLE
        return max(scores, key=lambda k: scores[k])


class RoleShardRouter:
    """Routes nodes and queries to role-based shards.

    Within each shard: full Kuramoto synchronization, full consolidation.
    Between shards: exchange only when cross_role_resonance > threshold.
    """

    def __init__(
        self, shards: Optional[Set[str]] = None, cross_shard_threshold: float = 0.45, auto_role_detection: bool = True
    ):
        self.cross_shard_threshold = cross_shard_threshold
        self.auto_role_detection = auto_role_detection
        self.role_detector = RoleDetector()
        self.shards: Dict[str, RoleShard] = {}
        self.node_role_map: Dict[str, str] = {}  # node_id → role

        # Initialize default shards
        for role in shards or {DEFAULT_ROLE}:
            self.shards[role] = RoleShard(role=role)

    def add_node(self, node_id: str, text: str, role: Optional[str] = None) -> str:
        """Add a node to the appropriate shard.

        Args:
            node_id: Node identifier
            text: Node text content (for auto-detection)
            role: Explicit role (overrides auto-detection)

        Returns:
            The role this node was assigned to
        """
        if role is None and self.auto_role_detection:
            role = self.role_detector.detect(text)
        elif role is None:
            role = DEFAULT_ROLE

        # Create shard if it doesn't exist
        if role not in self.shards:
            self.shards[role] = RoleShard(role=role)

        # Add node to shard
        self.shards[role].node_ids.add(node_id)
        self.node_role_map[node_id] = role
        return role

    def remove_node(self, node_id: str):
        """Remove a node from its shard."""
        role = self.node_role_map.pop(node_id, None)
        if role and role in self.shards:
            self.shards[role].node_ids.discard(node_id)
            self.shards[role].kuramoto_phases.pop(node_id, None)

    def get_node_role(self, node_id: str) -> str:
        """Get the role of a node."""
        return self.node_role_map.get(node_id, DEFAULT_ROLE)

    def get_relevant_shards(self, query_text: str, top_n: int = 2) -> List[str]:
        """Get the most relevant shards for a query.

        Returns top_n shard roles ordered by relevance.
        """
        if len(self.shards) <= 1:
            return list(self.shards.keys())

        # Score each shard by keyword match
        shard_scores: Dict[str, float] = {}
        for role, shard in self.shards.items():
            if not shard.node_ids:
                continue
            # Check if query matches shard role keywords
            detected_role = self.role_detector.detect(query_text)
            if detected_role == role:
                shard_scores[role] = 1.0
            elif role == DEFAULT_ROLE:
                # Default shard is always relevant fallback
                shard_scores[role] = 0.5
            else:
                shard_scores[role] = 0.1

        # Sort by score descending
        sorted_roles = sorted(shard_scores, key=lambda k: shard_scores[k], reverse=True)
        return sorted_roles[:top_n]

    def should_exchange(self, shard_a: str, shard_b: str, resonance_score: float) -> bool:
        """Check if two shards should exchange information.

        Exchange happens only when cross_role_resonance > threshold.
        """
        if shard_a == shard_b:
            return True  # Same shard, always exchange
        return resonance_score > self.cross_shard_threshold

    def update_kuramoto_phases(self, nodes: Dict[str, Any]):
        """Update Kuramoto phases within each shard.

        Should be called from RTMDKField.step().
        """
        for role, shard in self.shards.items():
            if len(shard.node_ids) < 2:
                continue

            # Collect phases from nodes in this shard
            new_phases: Dict[str, float] = {}
            for nid in shard.node_ids:
                if nid in nodes:
                    node = nodes[nid]
                    new_phases[nid] = getattr(node, "phase", 0.0)

            if len(new_phases) < 2:
                continue

            # Simple Kuramoto step within shard
            n = len(new_phases)
            K_over_N = shard.kuramoto_coupling / n
            updated_phases = {}
            for nid, phi in new_phases.items():
                coupling = 0.0
                for other_id, other_phi in new_phases.items():
                    if other_id != nid:
                        coupling += np.sin(other_phi - phi)
                updated_phases[nid] = (phi + 0.01 * K_over_N * coupling) % (2 * np.pi)

            # Apply phase updates
            for nid, new_phi in updated_phases.items():
                if nid in nodes:
                    nodes[nid].phase = float(new_phi)

            shard.kuramoto_phases = updated_phases
            shard.last_sync_time = time.time()

    def get_shard_for_query(self, query_text: str) -> str:
        """Get the primary shard for a query."""
        relevant = self.get_relevant_shards(query_text, top_n=1)
        return relevant[0] if relevant else DEFAULT_ROLE

    def get_stats(self) -> Dict:
        """Get statistics about all shards."""
        return {
            "n_shards": len(self.shards),
            "shards": {
                role: {
                    "n_nodes": len(shard.node_ids),
                    "n_queries": shard.n_queries,
                    "n_consolidations": shard.n_consolidations,
                    "n_cross_shard_exchanges": shard.n_cross_shard_exchanges,
                }
                for role, shard in self.shards.items()
            },
            "cross_shard_threshold": self.cross_shard_threshold,
        }

    def get_state(self) -> Dict:
        """Export state for serialization."""
        return {
            "shards": {role: shard.to_dict() for role, shard in self.shards.items()},
            "node_role_map": dict(self.node_role_map),
            "cross_shard_threshold": self.cross_shard_threshold,
            "auto_role_detection": self.auto_role_detection,
        }

    def load_state(self, data: Dict):
        """Import state from serialization."""
        self.shards = {}
        for role, shard_data in data.get("shards", {}).items():
            self.shards[role] = RoleShard.from_dict(shard_data)
        self.node_role_map = dict(data.get("node_role_map", {}))
        self.cross_shard_threshold = data.get("cross_shard_threshold", 0.45)
        self.auto_role_detection = data.get("auto_role_detection", True)
