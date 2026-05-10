from __future__ import annotations

"""
rtmdk/memory/serialization.py — Field serialization utilities.

Centralizes export/import logic to eliminate duplication between
RTMDKField.export_field, export_to_dict, import_field, import_from_dict.

Usage:
    from rtmdk.memory.serialization import FieldSerializer
    data = FieldSerializer.field_to_dict(field)
    FieldSerializer.field_to_file(field, "memory.json")
    field = FieldSerializer.field_from_file("memory.json", embedder, config)
"""

import hashlib
import os
import json
import logging
from typing import Dict, Any, Callable, Optional
from enum import Enum
import numpy as np


def _normalize_for_checksum(obj: Any) -> Any:
    """Recursively normalize data for stable checksum computation.

    Converts numpy arrays / sets → lists, ensures sortable keys.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, set):
        return sorted(_normalize_for_checksum(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _normalize_for_checksum(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_normalize_for_checksum(v) for v in obj]
    return obj

from rtmdk.memory.config import (
    ConsolidationMode, Backend, EvalMode, ContextFormat,
)
from rtmdk.memory.core import (
    RTMDKConfig, RTMDKField, MemoryNode,
)

logger = logging.getLogger("rtmdk.serialization")


class EnumSerializer:
    """Serialize enum values to strings for JSON compatibility."""

    @staticmethod
    def enum_to_value(val: Any, default: Any) -> Any:
        """Convert enum to its value, or return default."""
        return val.value if isinstance(
            val, Enum) else (
            val if val is not None else default)

    @staticmethod
    def serialize_config(cd: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize all enum fields in config dict."""
        cd["consolidation_mode"] = EnumSerializer.enum_to_value(
            cd.get("consolidation_mode"), "dialectical")
        cd["backend"] = EnumSerializer.enum_to_value(
            cd.get("backend"), "numpy")
        cd["context_format"] = EnumSerializer.enum_to_value(
            cd.get("context_format"), "plain")
        cd["eval_mode"] = EnumSerializer.enum_to_value(
            cd.get("eval_mode"), "production")
        if "memory_tiers" in cd and isinstance(cd["memory_tiers"], set):
            cd["memory_tiers"] = list(cd["memory_tiers"])
        return cd

    @staticmethod
    def deserialize_config(cd: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize string values back to enums."""
        if isinstance(cd.get("consolidation_mode"), str):
            cd["consolidation_mode"] = ConsolidationMode(
                cd["consolidation_mode"])
        if isinstance(cd.get("backend"), str):
            cd["backend"] = Backend(cd["backend"])
        if isinstance(cd.get("context_format"), str):
            cd["context_format"] = ContextFormat(cd["context_format"])
        if isinstance(cd.get("eval_mode"), str):
            cd["eval_mode"] = EvalMode(cd["eval_mode"])
        if "memory_tiers" in cd and isinstance(cd["memory_tiers"], list):
            cd["memory_tiers"] = set(cd["memory_tiers"])
        return cd


class FieldSerializer:
    """Serialize/deserialize RTMDKField state."""

    # Submodules that support get_state()/load_state()
    STATE_MODULES = [
        ("projection_learner", "projection_state", "learn_projection"),
        ("learnable_kernel", "learnable_kernel", "differentiable"),
        ("meta_kernel", "meta_kernel", "meta_adaptive"),
        ("healer", "healer", "self_healing"),
        ("causal_engine", "causal_engine", "causal_topological"),
        ("ode_dynamics", "ode_dynamics", "continuous_dynamics"),
        ("meta_controller", "meta_controller", "meta_controller"),
        ("federated", "federated", "federated"),
        ("meta_memory_eval", "meta_memory_eval", "meta_memory"),
        ("security", "security", "security_enabled"),
        ("version_control", "version_control", "version_control"),
        ("sot_tokenizer", "sot_tokenizer", "sot_enabled"),
    ]

    # Additional subsystems that use get_state()/load_state() but have no
    # dedicated config flag (checked by truthiness).
    EXTRA_STATE_MODULES = [
        ("learned_consolidator", "learned_consolidator"),
        ("event_scheduler", "event_scheduler"),
        ("low_rank_compressor", "low_rank_compressor"),
        ("goal_tracker", "goal_tracker"),
        ("rl_feedback_loop", "rl_feedback_loop"),
        ("predictor", "predictor"),
        ("scenario_planner", "scenario_planner"),
        ("engram_manager", "engram_manager"),
    ]

    @staticmethod
    def field_to_dict(field: RTMDKField) -> Dict[str, Any]:
        """Export field state to a dict."""
        # Serialize config
        cd = field.config.asdict() if hasattr(field, 'config') else field.cfg.asdict()
        cd = EnumSerializer.serialize_config(cd)

        # Build data dict
        data = {
            "_schema_version": "1.0",
            "config": cd,
            "nodes": list(
                field.nodes.all_node_dicts()) if hasattr(
                field.nodes,
                "all_node_dicts") else [
                n.to_dict() for n in field.nodes.values()],
            "stats": field.stats,
        }

        # Add projection + SOT state
        data.update(field._projection_mgr.get_state())

        # Add submodule states
        for attr, key, config_flag in FieldSerializer.STATE_MODULES:
            obj = getattr(field, attr, None)
            if obj is not None and hasattr(obj, 'get_state'):
                try:
                    data[key] = obj.get_state()
                except Exception as e:
                    logger.warning(f"Failed to serialize {attr}: {e}")

        # Add extra subsystem states (no config flag guard)
        for attr, key in FieldSerializer.EXTRA_STATE_MODULES:
            obj = getattr(field, attr, None)
            if obj is not None and hasattr(obj, 'get_state'):
                try:
                    data[key] = obj.get_state()
                except Exception as e:
                    logger.warning(f"Failed to serialize {attr}: {e}")

        # TDA history
        if field.tda_monitor and hasattr(field.tda_monitor, 'history'):
            data["tda_history"] = field.tda_monitor.history

        if field._tiered_store is not None:
            data["tiered_store"] = field._tiered_store.save_state()
        return data

    @staticmethod
    def field_to_file(field: RTMDKField, path: str, fmt: str = "json"):
        """Export field state to file."""
        # Path sanitization
        path = os.path.normpath(str(path))
        if ".." in path.split(os.sep):
            raise ValueError(
                f"Invalid path: path traversal not allowed: {path}")
        if not path.endswith((".json", ".msgpack")):
            raise ValueError(
                f"Invalid format: path must end with .json or .msgpack: {path}")

        data = FieldSerializer.field_to_dict(field)
        # v8.2.1: Snapshot integrity checksum
        data["_checksum"] = hashlib.sha256(
            json.dumps(_normalize_for_checksum(data), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        if fmt == "msgpack":
            try:
                import msgpack
                import zlib

                def _msgpack_default(obj):
                    if isinstance(obj, set):
                        return list(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    if isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    raise TypeError(f"Cannot serialize {type(obj)}")

                packed = msgpack.packb(
                    data, use_bin_type=True, default=_msgpack_default)
                compressed = zlib.compress(packed)
                tmp_path = path + ".tmp"
                with open(tmp_path, "wb") as f:
                    f.write(compressed)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except ImportError:
                import warnings
                warnings.warn("msgpack not installed, falling back to JSON")
                tmp_path = path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        data,
                        f,
                        ensure_ascii=False,
                        indent=2,
                        default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
        else:
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass  # Windows may not support chmod

    @staticmethod
    def field_from_file(path: str, embedder: Callable,
                        config: Optional[RTMDKConfig] = None,
                        wal_path: Optional[str] = None):
        """Import field state from file.

        Args:
            path: File path
            embedder: Embedding function
            config: Optional config override (uses file config if None)

        Returns:
            RTMDKMemory instance with loaded field
        """
        from rtmdk.memory.core import RTMDKMemory

        logger.info(f"import_field: loading from {path}")

        # Path sanitization
        path = os.path.normpath(str(path))
        if ".." in path.split(os.sep):
            raise ValueError(
                f"Invalid path: path traversal not allowed: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        file_size = os.path.getsize(path)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB (max 100MB)")
        if file_size < 10:
            raise ValueError(
                f"File too small ({file_size} bytes): possibly corrupted")

        # Auto-detect format
        with open(path, "rb") as f:
            header = f.read(2)
        is_msgpack = header[0:1] == b'\x78' and header[1:2] in (
            b'\x01', b'\x5e', b'\x9c', b'\xda')

        if is_msgpack:
            logger.info("import_field: detected msgpack+zlib format")
            try:
                import msgpack
                import zlib
                with open(path, "rb") as f:
                    compressed = f.read()
                packed = zlib.decompress(compressed)
                data = msgpack.unpackb(packed, raw=False)
            except ImportError:
                raise ImportError(
                    "msgpack required for binary import. Install: pip install msgpack")
        else:
            logger.info("import_field: loading JSON format")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        return FieldSerializer._apply_data(
            data, embedder, config=config, wal_path=wal_path,
            file_size=file_size)

    @staticmethod
    def field_from_dict(data: Dict[str, Any], embedder: Callable,
                        config: Optional[RTMDKConfig] = None,
                        wal_path: Optional[str] = None):
        """Import field state from an in-memory dict.

        Args:
            data: Dict with field state (same schema as field_to_dict output)
            embedder: Embedding function
            config: Optional config override
            wal_path: Optional WAL path

        Returns:
            RTMDKMemory instance with loaded field
        """
        return FieldSerializer._apply_data(
            data, embedder, config=config, wal_path=wal_path)

    @staticmethod
    def _apply_data(data: Dict[str, Any], embedder: Callable,
                    config: Optional[RTMDKConfig] = None,
                    wal_path: Optional[str] = None,
                    file_size: Optional[int] = None):
        """Shared import logic used by field_from_file and field_from_dict."""
        from rtmdk.memory.core import RTMDKMemory

        # v8.2.1: Snapshot integrity verification
        stored_checksum = data.pop("_checksum", None)
        if stored_checksum is not None:
            computed = hashlib.sha256(
                json.dumps(_normalize_for_checksum(data), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if computed != stored_checksum:
                raise ValueError(
                    f"Corrupted snapshot: checksum mismatch (expected {stored_checksum}, got {computed})"
                )
            logger.info("import_field: checksum verified")

        # Health check
        if "config" not in data:
            raise ValueError("Invalid memory file: missing 'config' key")
        if "nodes" not in data:
            raise ValueError("Invalid memory file: missing 'nodes' key")

        n_file_nodes = len(data["nodes"])
        size_msg = f", {file_size/1024:.0f}KB" if file_size else ""
        logger.info(
            f"import_field: loading {n_file_nodes} nodes{size_msg}")

        # Deserialize config
        cd = data["config"]
        cd = EnumSerializer.deserialize_config(cd)

        # Handle v5/v6 backward compatibility
        if "causal_modeling" in cd and "causal_topological" not in cd:
            cd["causal_topological"] = cd.pop("causal_modeling")
        elif "causal_modeling" in cd:
            cd.pop("causal_modeling")

        valid_fields = set(
            f.name for f in RTMDKConfig.__dataclass_fields__.values())
        cd = {k: v for k, v in cd.items() if k in valid_fields}

        if config is None:
            config = RTMDKConfig(**cd)

        logger.info(
            f"import_field: loading config (context_format={cd.get('context_format', '?')})")

        memory = RTMDKMemory(
            config=config,
            embedder=embedder,
            wal_path=wal_path)

        # Load projection + SOT state
        memory.field._projection_mgr.load_state(data)

        # Load submodule states
        if config.differentiable and "learnable_kernel" in data:
            memory.field.learnable_kernel.load_state(data["learnable_kernel"])
        if config.tda_monitoring and "tda_history" in data:
            memory.field.tda_monitor.history = data["tda_history"]
        if config.meta_adaptive and "meta_kernel" in data:
            memory.field.meta_kernel.load_state(data["meta_kernel"])
        if config.self_healing and "healer" in data:
            memory.field.healer.load_state(data["healer"])
        if config.causal_topological and "causal_engine" in data:
            memory.field.causal_engine.load_state(data["causal_engine"])
        if config.continuous_dynamics and "ode_dynamics" in data:
            ode_state = data["ode_dynamics"]
            memory.field.ode_dynamics.alpha = ode_state.get("alpha", 0.1)
            memory.field.ode_dynamics.beta = ode_state.get("beta", 0.05)
            memory.field.ode_dynamics.gamma = ode_state.get("gamma", 0.02)
            if "W" in ode_state:
                memory.field.ode_dynamics.W = np.array(
                    ode_state["W"], dtype=np.float32)
            memory.field.ode_dynamics.noise_level = ode_state.get(
                "noise_level", 0.01)
        if config.meta_controller and "meta_controller" in data:
            memory.field.meta_controller.load_state(data["meta_controller"])
        if config.federated and "federated" in data:
            memory.field.federated.import_state(data["federated"])
        if config.meta_memory and "meta_memory_eval" in data:
            memory.field.meta_memory_eval.load_state(data["meta_memory_eval"])
        if config.security_enabled and "security" in data:
            memory.field.security.load_state(data["security"])
        if getattr(config, "learned_consolidation", False) and "learned_consolidator" in data and memory.field.learned_consolidator is not None:
            memory.field.learned_consolidator.load_state(data["learned_consolidator"])

        # Load extra subsystem states (checked by attribute truthiness)
        for attr, key in FieldSerializer.EXTRA_STATE_MODULES:
            if key in data and getattr(memory.field, attr, None) is not None:
                obj = getattr(memory.field, attr)
                if hasattr(obj, 'load_state'):
                    obj.load_state(data[key])
                elif hasattr(obj, 'import_state'):
                    obj.import_state(data[key])

        # Load nodes
        logger.info(f"import_field: loading {len(data['nodes'])} nodes")
        for nd in data["nodes"]:
            node = MemoryNode.from_dict(nd)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        if "tiered_store" in data and memory.field._tiered_store is not None:
            memory.field._tiered_store.load_state(data["tiered_store"])
        logger.info(
            f"import_field: successfully loaded {len(memory.field.nodes)} nodes")

        # Reconcile stats
        saved_stats = data.get("stats", {})
        memory.field.stats = saved_stats
        n_nodes = len(memory.field.nodes)
        logger.info(f"import_field: reconciling stats for {n_nodes} nodes")
        memory.field.stats["total_adds"] = n_nodes
        memory.field.stats["active_nodes"] = n_nodes

        # Reset historical accumulation counters
        reset_keys = [
            "projection_updates", "self_sup_checks", "total_queries",
            "consolidations", "consolidation_validations", "blocked_consolidations",
            "healing_events", "healing_history", "field_stability",
            "tension_cache_hits", "tension_cache_misses", "tension_cache_hit_rate",
            "engram_retrievals", "engrams_created", "engrams_merged",
            "cross_modal_queries", "cross_modal_recall",
            "meta_optimizations", "meta_best_params",
            "federated_syncs", "federated_order_parameter",
            "crystallizations", "crystallized_clusters",
            "evaluations", "shadow_comparisons", "rollbacks",
            "ode_steps", "response_smoothness",
            "free_energy", "prediction_error", "surprise_level",
            "scenarios_generated", "avg_scenario_confidence",
            "privacy_budget_spent", "noise_std", "updates_clipped",
            "shard_hits", "shard_misses", "avg_shard_query_time_ms",
            "context_tokens_saved", "cognitive_compressions",
            "async_queue_depth", "async_backpressure_events",
            "active_goals", "completed_goals",
            "avg_rl_reward", "reward_trend",
            "attention_bias_applied", "compression_ratio", "compression_updates",
            "events_processed", "event_queue_depth",
            "recall_accuracy", "meta_reflections",
            "security_violations", "tension_spikes_blocked",
            "current_version", "n_versions",
            "clarifications_generated",
            "plans_created", "hypotheses_verified", "tool_calls", "tool_misuse_rate",
            "ragas_overall", "tier_coherence",
            "n_symbolic_rules", "n_symbolic_inferences", "n_symbolic_conflicts",
            "lyapunov_V", "lyapunov_dV_dt", "safety_regulation_factor", "safety_mode",
            "n_shards", "shard_distribution", "cross_shard_exchanges",
            "role_router_enabled",
            "field_integrity_issues",
            "hybrid_retrievals",
        ]
        for key in reset_keys:
            if key in memory.field.stats:
                val = memory.field.stats[key]
                if isinstance(val, (int, float)):
                    memory.field.stats[key] = 0
                elif isinstance(val, dict):
                    memory.field.stats[key] = {}
                elif isinstance(val, list):
                    memory.field.stats[key] = []

        # Recalculate tier_distribution from actual nodes
        tier_dist = {}
        for node in memory.field.nodes.values():
            tier = node.content.get(
                "tier", node.tier if hasattr(
                    node, 'tier') else "semantic")
            tier_dist[tier] = tier_dist.get(tier, 0) + 1
        memory.field.stats["tier_distribution"] = tier_dist
        memory.field.stats["avg_response"] = 0.0

        logger.info(
            f"import_field: complete — {n_nodes} nodes, tier_distribution={tier_dist}")
        return memory
