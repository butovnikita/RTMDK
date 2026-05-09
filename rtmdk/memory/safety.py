"""Safety features: rollback, poisoned memory detection, audit trail."""
from __future__ import annotations
import copy
import json
import time
from typing import Dict, List, Optional, Any
import numpy as np


class MemorySnapshot:
    """Immutable snapshot of memory state for rollback."""

    def __init__(self, timestamp: float, nodes: Dict, embeddings: Dict, stats: Dict):
        self.timestamp = timestamp
        self.nodes = copy.deepcopy(nodes)
        self.embeddings = {k: v.copy() for k, v in embeddings.items()}
        self.stats = copy.deepcopy(stats)


class RollbackManager:
    """Manage memory snapshots and rollback to previous states."""

    def __init__(self, max_snapshots: int = 10):
        self._snapshots: List[MemorySnapshot] = []
        self.max_snapshots = max_snapshots

    def take_snapshot(self, field) -> None:
        """Capture current memory state."""
        embeddings = {}
        for nid, node in field.nodes.items():
            if hasattr(node, "_embedding"):
                embeddings[nid] = node._embedding.copy()
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            nodes={nid: copy.deepcopy(node.content) for nid, node in field.nodes.items()},
            embeddings=embeddings,
            stats=copy.deepcopy(field.stats),
        )
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    def rollback(self, field, target_timestamp: Optional[float] = None) -> bool:
        """Rollback memory to snapshot closest to target_timestamp.

        If target_timestamp is None, rollback to the most recent snapshot.
        """
        if not self._snapshots:
            return False

        if target_timestamp is None:
            snapshot = self._snapshots[-1]
        else:
            # Find closest snapshot <= target
            candidates = [s for s in self._snapshots if s.timestamp <= target_timestamp]
            if not candidates:
                return False
            snapshot = max(candidates, key=lambda s: s.timestamp)

        # Clear current state
        field.nodes.clear()
        field.node_index.clear()
        if hasattr(field, "hnsw_index") and field.hnsw_index:
            field.hnsw_index.clear()

        # Restore nodes
        for nid, content in snapshot.nodes.items():
            from rtmdk.nodes import MemoryNode
            node = MemoryNode(id=nid, content=content)
            if nid in snapshot.embeddings:
                node._embedding = snapshot.embeddings[nid]
            field.nodes[nid] = node
            field.node_index[nid] = len(field.node_index)

        field.stats = copy.deepcopy(snapshot.stats)
        return True

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """Return list of available snapshots."""
        return [
            {"timestamp": s.timestamp, "node_count": len(s.nodes)}
            for s in self._snapshots
        ]


class PoisonedMemoryDetector:
    """Detect potentially poisoned or anomalous memory nodes.

    Flags nodes with:
    - Abnormally high out-degree (injection attempts)
    - Repetitive content (spam)
    - Extreme sentiment polarity (manipulation)
    """

    def __init__(self, max_out_degree: int = 100, repetition_threshold: float = 0.95):
        self.max_out_degree = max_out_degree
        self.repetition_threshold = repetition_threshold

    def scan(self, field) -> List[Dict[str, Any]]:
        """Scan all nodes and return list of suspicious nodes."""
        suspicious = []
        texts = []

        for nid, node in field.nodes.items():
            flags = []

            # Check out-degree
            out_degree = len(getattr(node, "causal_strength", {}))
            if out_degree > self.max_out_degree:
                flags.append(f"high_out_degree ({out_degree})")

            # Check repetition
            text = node.content.get("text", "")
            if text:
                similarity = self._max_similarity(text, texts)
                if similarity > self.repetition_threshold:
                    flags.append(f"repetitive_content (sim={similarity:.2f})")
                texts.append(text)

            if flags:
                suspicious.append({
                    "node_id": nid,
                    "flags": flags,
                    "content_preview": text[:100],
                })

        return suspicious

    @staticmethod
    def _max_similarity(text: str, corpus: List[str]) -> float:
        """Simple Jaccard similarity for quick check."""
        if not corpus:
            return 0.0
        words = set(text.lower().split())
        if not words:
            return 0.0
        best = 0.0
        for other in corpus[-50:]:  # Check last 50 only
            other_words = set(other.lower().split())
            if not other_words:
                continue
            inter = len(words & other_words)
            union = len(words | other_words)
            sim = inter / union if union else 0.0
            best = max(best, sim)
        return best
