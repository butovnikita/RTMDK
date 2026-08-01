"""rtmdk/support/version_control.py — Delta-based version control for memory fields."""

from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import numpy as np


@dataclass
class NodeDelta:
    """Delta for a single node change."""

    node_id: str
    action: str  # "added", "modified", "deleted", "merged"
    old_state: Optional[Dict] = None
    new_state: Optional[Dict] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        for k in ("old_state", "new_state"):
            if d[k] is not None:
                v = d[k]
                for kk in (
                    "latent_pos",
                    "pre_consolidation_pos",
                    "velocity",
                    "acceleration",
                    "gradient_cache",
                    "modal_embedding",
                ):
                    if kk in v and isinstance(v[kk], np.ndarray):
                        v[kk] = v[kk].tolist()
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "NodeDelta":
        for k in ("old_state", "new_state"):
            if data.get(k):
                v = data[k]
                for kk in (
                    "latent_pos",
                    "pre_consolidation_pos",
                    "velocity",
                    "acceleration",
                    "gradient_cache",
                    "modal_embedding",
                ):
                    if kk in v and isinstance(v[kk], list):
                        v[kk] = np.array(v[kk], dtype=np.float32)
        return cls(**data)


@dataclass
class Version:
    """A single version (commit) of the memory field."""

    version_id: int
    timestamp: float = field(default_factory=time.time)
    deltas: List[NodeDelta] = field(default_factory=list)
    message: str = ""
    parent_id: Optional[int] = None
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "version_id": self.version_id,
            "timestamp": self.timestamp,
            "deltas": [d.to_dict() for d in self.deltas],
            "message": self.message,
            "parent_id": self.parent_id,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Version":
        return cls(
            version_id=data["version_id"],
            timestamp=data["timestamp"],
            deltas=[NodeDelta.from_dict(d) for d in data["deltas"]],
            message=data.get("message", ""),
            parent_id=data.get("parent_id"),
            stats=data.get("stats", {}),
        )


@dataclass
class DiffResult:
    """Result of diff between two versions."""

    from_version: int
    to_version: int
    added_nodes: List[str] = field(default_factory=list)
    deleted_nodes: List[str] = field(default_factory=list)
    modified_nodes: List[str] = field(default_factory=list)
    merged_nodes: List[str] = field(default_factory=list)
    total_delta_size: int = 0  # Number of nodes changed

    def summary(self) -> str:
        lines = [f"Diff v{self.from_version} → v{self.to_version}:"]
        if self.added_nodes:
            lines.append(
                f"  + {len(self.added_nodes)} added: "
                f"{self.added_nodes[:5]}"
                f"{'...' if len(self.added_nodes) > 5 else ''}"
            )
        if self.deleted_nodes:
            lines.append(f"  - {len(self.deleted_nodes)} deleted")
        if self.modified_nodes:
            lines.append(f"  ~ {len(self.modified_nodes)} modified")
        if self.merged_nodes:
            lines.append(f"  ⨝ {len(self.merged_nodes)} merged")
        if not any([self.added_nodes, self.deleted_nodes, self.modified_nodes, self.merged_nodes]):
            lines.append("  (no changes)")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return asdict(self)


class VersionControl:
    """Delta-based version control for RTMDK memory fields.

    Stores only changed nodes (deltas) between versions, enabling:
    - Efficient history traversal (O(Δ) per version instead of O(N))
    - Diff between any two versions
    - Full rollback to any version
    - Per-node rollback (rollback specific nodes to earlier state)
    """

    def __init__(self, max_versions: int = 100):
        self.max_versions = max_versions
        self._versions: OrderedDict[int, Version] = OrderedDict()
        self._current_version: int = 0
        # node_id → {version_id → state}
        self._node_states: Dict[str, Dict[int, Dict]] = {}

    def create_version(self, deltas: List[NodeDelta], message: str = "", stats: Optional[Dict] = None) -> Version:
        """Create a new version from deltas."""
        self._current_version += 1
        version = Version(
            version_id=self._current_version,
            deltas=deltas,
            message=message or f"Auto v{self._current_version}",
            parent_id=self._current_version - 1 if self._current_version > 1 else None,
            stats=stats or {},
        )
        self._versions[version.version_id] = version

        # Track per-node states
        for delta in deltas:
            if delta.node_id not in self._node_states:
                self._node_states[delta.node_id] = {}
            if delta.new_state is not None:
                self._node_states[delta.node_id][version.version_id] = delta.new_state

        # Prune old versions
        while len(self._versions) > self.max_versions:
            oldest_id = next(iter(self._versions))
            del self._versions[oldest_id]

        return version

    def diff(self, from_version: int, to_version: Optional[int] = None) -> DiffResult:
        """Compute diff between two versions."""
        if to_version is None:
            to_version = self._current_version
        if from_version not in self._versions or to_version not in self._versions:
            return DiffResult(from_version=from_version, to_version=to_version)

        result = DiffResult(from_version=from_version, to_version=to_version)

        # Collect all deltas between versions
        version_ids = list(self._versions.keys())
        start_idx = version_ids.index(from_version) if from_version in version_ids else -1
        end_idx = version_ids.index(to_version) if to_version in version_ids else -1
        if start_idx < 0 or end_idx < 0:
            return result

        seen_nodes: Dict[str, str] = {}  # node_id → last action
        for vid in version_ids[start_idx + 1 : end_idx + 1]:
            version = self._versions[vid]
            for delta in version.deltas:
                prev_action = seen_nodes.get(delta.node_id)
                if delta.action == "added":
                    if prev_action != "added" or vid != version_ids[start_idx + 1]:
                        result.added_nodes.append(delta.node_id)
                elif delta.action == "deleted":
                    result.deleted_nodes.append(delta.node_id)
                elif delta.action == "modified":
                    if prev_action != "added":
                        result.modified_nodes.append(delta.node_id)
                elif delta.action == "merged":
                    result.merged_nodes.append(delta.node_id)
                seen_nodes[delta.node_id] = delta.action

        result.total_delta_size = len(
            set(result.added_nodes + result.deleted_nodes + result.modified_nodes + result.merged_nodes)
        )
        return result

    def rollback_to(self, version_id: int) -> List[NodeDelta]:
        """Generate rollback deltas to restore a specific version."""
        if version_id not in self._versions:
            return []

        # Compute forward deltas from current to target
        result = self.diff(version_id, self._current_version)

        rollback_deltas = []
        # Reverse deletions → add back
        for nid in result.deleted_nodes:
            state = self._get_node_state_at(nid, version_id)
            if state:
                rollback_deltas.append(NodeDelta(node_id=nid, action="added", new_state=state))
        # Reverse additions → delete
        for nid in result.added_nodes:
            rollback_deltas.append(NodeDelta(node_id=nid, action="deleted"))
        # Reverse modifications → restore old state
        for nid in result.modified_nodes:
            old_state = self._get_node_state_at(nid, version_id)
            if old_state:
                rollback_deltas.append(NodeDelta(node_id=nid, action="modified", old_state=None, new_state=old_state))

        return rollback_deltas

    def get_node_history(self, node_id: str) -> List[Tuple[int, Dict]]:
        """Get version history of a specific node."""
        if node_id not in self._node_states:
            return []
        return sorted(self._node_states[node_id].items())

    def history(self, limit: int = 20) -> List[Dict]:
        """Get version history."""
        items = list(self._versions.values())[-limit:]
        return [
            {
                "version": v.version_id,
                "timestamp": v.timestamp,
                "message": v.message,
                "n_deltas": len(v.deltas),
                "parent": v.parent_id,
            }
            for v in items
        ]

    def get_version_stats(self, version_id: int) -> Optional[Dict]:
        """Get stats for a specific version."""
        if version_id in self._versions:
            return self._versions[version_id].stats
        return None

    def export_state(self) -> Dict:
        """Export version control state for serialization."""
        return {
            "current_version": self._current_version,
            "max_versions": self.max_versions,
            "versions": {str(k): v.to_dict() for k, v in self._versions.items()},
            "node_states": {
                nid: {str(vid): state for vid, state in states.items()} for nid, states in self._node_states.items()
            },
        }

    def import_state(self, data: Dict):
        """Import version control state."""
        self._current_version = data.get("current_version", 0)
        self.max_versions = data.get("max_versions", self.max_versions)
        self._versions = OrderedDict()
        for k, v in data.get("versions", {}).items():
            self._versions[int(k)] = Version.from_dict(v)
        self._node_states = {}
        for nid, states in data.get("node_states", {}).items():
            self._node_states[nid] = {int(vid): state for vid, state in states.items()}

    def _get_node_state_at(self, node_id: str, version_id: int) -> Optional[Dict]:
        """Get node state at a specific version."""
        if node_id not in self._node_states:
            return None
        states = self._node_states[node_id]
        # Find the latest state ≤ version_id
        best_vid: Optional[int] = None
        for vid in sorted(states.keys()):
            if vid <= version_id:
                best_vid = vid
        if best_vid is None:
            return None
        return states.get(best_vid)

    @property
    def current_version(self) -> int:
        return self._current_version

    @property
    def n_versions(self) -> int:
        return len(self._versions)
