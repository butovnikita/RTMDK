"""rtmdk/support/ump.py — Universal Memory Protocol v1 for RTMDK.

Standardized exchange format with:
- Schema versioning
- SHA-256 integrity check
- Topology metadata
- Consolidation log
- Kernel parameters

Phase 1: Schema + export/import (no semantic merge yet).
"""
from __future__ import annotations
import json
import hashlib
import time
from typing import Dict, Any
from dataclasses import dataclass, field, asdict


UMP_VERSION = "1.0.0"
UMP_SCHEMA = "rtmdk-v8"


@dataclass
class UMPHeader:
    """Metadata header for UMP export."""
    ump_version: str = UMP_VERSION
    schema: str = UMP_SCHEMA
    timestamp: float = field(default_factory=time.time)
    sha256: str = ""
    source: str = ""  # Optional: source identifier (device, agent, etc.)
    comment: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "UMPHeader":
        return cls(**{k: v for k, v in data.items()
                   if k in cls.__dataclass_fields__})


@dataclass
class UMPTopology:
    """Summary of field topology."""
    n_nodes: int = 0
    n_causal_edges: int = 0
    n_contradictions: int = 0
    latent_dim: int = 64
    embedding_dim: int = 768
    kernel: str = "gaussian_phase"
    n_tiers: Dict[str, int] = field(default_factory=dict)
    modalities: Dict[str, int] = field(default_factory=dict)
    field_health: str = "stable"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "UMPTopology":
        return cls(**{k: v for k, v in data.items()
                   if k in cls.__dataclass_fields__})


@dataclass
class UMPKernelParams:
    """Kernel parameters for reproducibility."""
    bandwidth: float = 1.0
    phase_coupling: float = 0.3
    decay_rate: float = 0.998
    tension_threshold: float = 0.25
    min_response: float = 0.1
    top_k: int = 5
    context_format: str = "plain"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "UMPKernelParams":
        return cls(**{k: v for k, v in data.items()
                   if k in cls.__dataclass_fields__})


def compute_sha256(data: Dict) -> str:
    """Compute SHA-256 hash of a dict (for integrity check)."""
    serialized = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def verify_sha256(data: Dict, expected_hash: str) -> bool:
    """Verify SHA-256 hash of data (excluding the hash field itself)."""
    data_without_hash = {k: v for k, v in data.items() if k != "sha256"}
    computed = compute_sha256(data_without_hash)
    return computed == expected_hash


class UniversalMemoryProtocol:
    """UMP v1: Schema + integrity check + export/import."""

    @staticmethod
    def export(
            field: Any,
            memory: Any,
            source: str = "",
            comment: str = "") -> Dict:
        """Export RTMDK memory to UMP format.

        Args:
            field: RTMDKField instance
            memory: RTMDKMemory instance
            source: Optional source identifier
            comment: Optional comment

        Returns:
            UMP dict ready for JSON serialization
        """
        # Build topology summary
        topology = UMPTopology(
            n_nodes=len(field.nodes),
            n_causal_edges=field.stats.get("causal_edges", 0),
            n_contradictions=field.stats.get("contradictions", 0),
            latent_dim=field.cfg.latent_dim,
            embedding_dim=field.cfg.embedding_dim,
            kernel=field.cfg.resonance_kernel,
            field_health=field.stats.get("field_health", "stable"),
        )

        # Tier distribution
        tier_dist: Dict[str, int] = {}
        modality_dist: Dict[str, int] = {}
        for node in field.nodes.values():
            tier = getattr(node, 'tier', 'semantic')
            tier_dist[tier] = tier_dist.get(tier, 0) + 1
            mod = getattr(node, 'modality', 'text')
            modality_dist[mod] = modality_dist.get(mod, 0) + 1
        topology.n_tiers = tier_dist
        topology.modalities = modality_dist

        # Kernel params
        kernel_params = UMPKernelParams(
            bandwidth=field.cfg.bandwidth,
            phase_coupling=field.cfg.phase_coupling,
            decay_rate=field.cfg.decay_rate,
            tension_threshold=field.cfg.tension_threshold,
            min_response=field.cfg.min_response,
            top_k=field.cfg.top_k,
            context_format=field.cfg.context_format.value,
        )

        # Consolidation log (from version control if available)
        consolidation_log = []
        if hasattr(field, 'version_control') and field.version_control:
            for v_info in field.version_control.history(limit=50):
                consolidation_log.append(v_info)

        # Build full export (using existing export mechanism)
        field_state = field.export_to_dict()

        # Assemble UMP
        ump = {
            "ump_version": UMP_VERSION,
            "schema": UMP_SCHEMA,
            "header": UMPHeader(source=source, comment=comment).to_dict(),
            "topology": topology.to_dict(),
            "kernel_params": kernel_params.to_dict(),
            "consolidation_log": consolidation_log,
            "field_state": field_state,
        }

        # Compute SHA-256 (excluding sha256 field itself)
        sha256 = compute_sha256(ump)
        ump["header"]["sha256"] = sha256

        return ump

    @staticmethod
    def import_ump(ump: Dict, embedder, memory_class=None) -> Any:
        """Import RTMDK memory from UMP format.

        Args:
            ump: UMP dict
            embedder: Embedder function
            memory_class: RTMDKMemory class (for circular import avoidance)

        Returns:
            RTMDKMemory instance
        """
        if memory_class is None:
            from rtmdk.memory.core import RTMDKMemory
            memory_class = RTMDKMemory

        # Verify SHA-256
        expected_hash = ump.get("header", {}).get("sha256", "")
        if expected_hash:
            if not verify_sha256(ump, expected_hash):
                raise ValueError(
                    "UMP integrity check failed: SHA-256 mismatch. "
                    f"Expected: {expected_hash[:16]}...")

        # Validate schema
        ump_ver = ump.get("ump_version", "unknown")
        schema = ump.get("schema", "unknown")
        if schema != UMP_SCHEMA:
            raise ValueError(
                f"UMP schema mismatch: expected {UMP_SCHEMA}, got {schema}. "
                f"UMP version: {ump_ver}")

        # Import field state
        field_state = ump.get("field_state", {})
        if not field_state:
            raise ValueError("UMP missing field_state")

        # Reconstruct memory
        memory = memory_class.import_from_dict(field_state, embedder)
        return memory

    @staticmethod
    def validate(ump: Dict) -> Dict[str, Any]:
        """Validate UMP format and integrity."""
        issues = []
        warnings = []

        # Check required fields
        required = [
            "ump_version",
            "schema",
            "header",
            "topology",
            "kernel_params",
            "field_state"]
        for field_name in required:
            if field_name not in ump:
                issues.append(f"Missing required field: {field_name}")

        # Check schema version
        if ump.get("schema") != UMP_SCHEMA:
            warnings.append(
                f"Schema version mismatch: expected {UMP_SCHEMA}, got {ump.get('schema')}")

        # Verify SHA-256
        if "header" in ump and "sha256" in ump["header"]:
            if not verify_sha256(ump, ump["header"]["sha256"]):
                issues.append("SHA-256 integrity check failed")
        else:
            warnings.append("No SHA-256 hash found in header")

        # Check topology consistency
        if "topology" in ump:
            topo = ump["topology"]
            if topo.get("n_nodes", 0) <= 0:
                warnings.append("Field has no nodes")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }

    @staticmethod
    def diff(base_state: Dict, new_state: Dict) -> Dict:
        """D2: Compute delta between two RTMDK field states.

        Args:
            base_state: Original field state dict (from export_to_dict)
            new_state: New field state dict

        Returns:
            Delta dict: {"added_nodes": [...], "removed_nodes": [...], "modified_nodes": [...]}
        """
        base_nodes = base_state.get("nodes", {})
        new_nodes = new_state.get("nodes", {})

        base_ids = set(base_nodes.keys())
        new_ids = set(new_nodes.keys())

        added_ids = new_ids - base_ids
        removed_ids = base_ids - new_ids
        common_ids = base_ids & new_ids

        added_nodes = []
        for nid in added_ids:
            node = new_nodes[nid]
            added_nodes.append({"id": nid, **node})

        removed_nodes = []
        for nid in removed_ids:
            node = base_nodes[nid]
            removed_nodes.append({"id": nid, **node})

        modified_nodes = []
        for nid in common_ids:
            base_node = base_nodes[nid]
            new_node = new_nodes[nid]
            if base_node != new_node:
                # Compute field-level diff
                changes = {}
                for key in set(list(base_node.keys()) + list(new_node.keys())):
                    old_val = base_node.get(key)
                    new_val = new_node.get(key)
                    if old_val != new_val:
                        changes[key] = {"old": old_val, "new": new_val}
                if changes:
                    modified_nodes.append({"id": nid, "changes": changes})

        return {
            "added_nodes": added_nodes,
            "removed_nodes": removed_nodes,
            "modified_nodes": modified_nodes,
            "delta_hash": compute_sha256({
                "added": len(added_nodes),
                "removed": len(removed_nodes),
                "modified": len(modified_nodes),
            }),
        }

    @staticmethod
    def apply_delta(state: Dict, delta: Dict) -> Dict:
        """D2: Apply a delta patch to a field state.

        Args:
            state: Current field state dict (from export_to_dict)
            delta: Delta dict from diff()

        Returns:
            Updated field state dict
        """
        import copy
        result = copy.deepcopy(state)
        nodes = result.get("nodes", {})

        # Remove nodes
        for removed in delta.get("removed_nodes", []):
            nid = removed.get("id") if isinstance(removed, dict) else removed
            nodes.pop(nid, None)

        # Add nodes
        for added in delta.get("added_nodes", []):
            nid = added.get("id")
            if nid:
                node_data = {k: v for k, v in added.items() if k != "id"}
                nodes[nid] = node_data

        # Modify nodes
        for modified in delta.get("modified_nodes", []):
            nid = modified.get("id")
            if nid and nid in nodes and "changes" in modified:
                for field_name, change in modified["changes"].items():
                    if isinstance(change, dict) and "new" in change:
                        nodes[nid][field_name] = change["new"]
                    else:
                        nodes[nid][field_name] = change

        result["nodes"] = nodes
        # Update topology
        if "topology" in result:
            result["topology"]["n_nodes"] = len(nodes)

        return result
