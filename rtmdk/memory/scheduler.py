"""StepScheduler — periodic maintenance task orchestrator.

Extracted from RTMDKField to reduce monolithic field.py size.
Delegates all actual work back to the parent field or its sub-managers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField

logger = logging.getLogger(__name__)


class StepScheduler:
    """Orchestrates periodic tasks executed on every step() call."""

    def __init__(self, field: RTMDKField) -> None:
        self.field = field

    def run(self, backpressure_ok: bool) -> None:
        """Execute all periodic maintenance tasks."""
        field = self.field
        cfg = field.cfg
        n_nodes = len(field.nodes)
        step = field._step_counter

        # Consolidation
        if n_nodes > 10:
            if n_nodes < 1000:
                consolidation_freq = 100
            elif n_nodes < 10000:
                consolidation_freq = 50
            else:
                consolidation_freq = 20
            if step % consolidation_freq == 0:
                if cfg.consolidation_async:
                    if field._consolidation_future is None or field._consolidation_future.done():
                        field._consolidation_future = field._consolidation_executor.submit(
                            field._circuit_breakers["Consolidate"].call, field.consolidate
                        )
                else:
                    field._circuit_breakers["Consolidate"].call(field.consolidate)

        # Self-healing
        if cfg.self_healing and step % cfg.healing_check_freq == 0:
            field._circuit_breakers["SelfHeal"].call(field._operational_mgr.self_heal)

        # Tier decay
        tier_counts: Dict[str, int] = {}
        tier_amplitudes: Dict[str, List[float]] = {}
        for node in field.nodes.values():
            tier = getattr(node, "tier", "semantic")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            dk = cfg.tier_decay.get(tier, cfg.decay_rate)
            if field.learnable_kernel:
                dk = max(dk, field.learnable_kernel.decay_rate)
            node.amplitude *= dk
            node.salience *= dk
            node.amplitude = float(np.clip(node.amplitude, cfg.min_amplitude, 1.0))
            node.salience = float(np.clip(node.salience, cfg.min_amplitude * 0.5, 1.0))
            tier_amplitudes.setdefault(tier, []).append(node.amplitude)
        field.stats["tier_distribution"] = tier_counts
        if tier_amplitudes:
            coherences = []
            for amps in tier_amplitudes.values():
                coherences.append(1.0 - np.std(amps) if len(amps) > 1 else 1.0)
            field.stats["tier_coherence"] = float(np.mean(coherences)) if coherences else 0.0

        # Predictive coding
        if field.predictor and field.nodes and step % 5 == 0:
            state = field._encode_field_state()
            field._state_history.append(state)
            if len(field._state_history) >= 2:
                fe = field._circuit_breakers["PredictorFreeEnergy"].call(
                    field.predictor.compute_free_energy, field._state_history[-2], field._state_history[-1]
                )
                field.stats["free_energy"] = fe
                field.stats["prediction_error"] = float(
                    np.mean((field.predictor.predict(field._state_history[-2]) - field._state_history[-1]) ** 2)
                )
                field.stats["surprise_level"] = float(np.clip(fe, 0, 1))
                if fe > 0.3 and n_nodes > 10:
                    field._circuit_breakers["Consolidate"].call(field.consolidate)
                if fe > 0.01:
                    field._circuit_breakers["PredictorUpdate"].call(
                        field.predictor.update, field._state_history[-2], field._state_history[-1], lr=cfg.pc_lr
                    )

        # Max-nodes pruning
        if cfg.max_nodes and n_nodes > cfg.max_nodes and step % 10 == 0:
            sorted_nodes = sorted(
                field.node_index, key=lambda nid: field.nodes[nid].salience * field.nodes[nid].amplitude
            )
            n_pruned = n_nodes - cfg.max_nodes
            pruned_ids = set(sorted_nodes[:n_pruned])
            if pruned_ids:
                field.wal.append_delete(list(pruned_ids))
            for nid in pruned_ids:
                if cfg.use_hnsw:
                    field._index_mgr.hnsw_remove(nid)
                if cfg.bm25_fallback:
                    field._index_mgr.bm25_remove(nid)
                del field.nodes[nid]
            field.node_index = [nid for nid in field.node_index if nid not in pruned_ids]
            if n_pruned > 0:
                field._invalidate_tension_cache()

        # Self-supervision
        if cfg.self_supervision and step % 20 == 0:
            field._circuit_breakers["SelfSupervise"].call(field._self_supervise)

        # TDA
        if backpressure_ok and cfg.tda_monitoring and step % cfg.tda_check_freq == 0:
            field._circuit_breakers["TDA"].call(field._check_tda)

        # Meta-kernel adaptation
        if field.meta_kernel and step % 5 == 0:
            field._circuit_breakers["MetaKernelAdapt"].call(field.meta_kernel.adapt)
            field.stats["meta_kurtosis"] = field.meta_kernel.compute_resonance_kurtosis()
            field.stats["meta_bandwidth"] = field.meta_kernel.get_bandwidth()
            field.stats["meta_phase_coupling"] = field.meta_kernel.get_phase_coupling()

        # Meta-controller
        if (
            backpressure_ok
            and field.meta_controller
            and field.meta_controller.should_optimize()
            and step % cfg.meta_opt_freq == 0
        ):
            best_params = field._circuit_breakers["MetaControllerOptimize"].call(field.meta_controller.optimize, field)
            if best_params:
                field._circuit_breakers["MetaControllerApply"].call(
                    field.meta_controller.apply_params, field, best_params
                )
                field.stats["meta_optimizations"] += 1
                field.stats["meta_best_params"] = best_params

        # Federated sync
        if field.federated and step > 0 and step % cfg.federated_sync_freq == 0:
            local_phases = {nid: n.phase for nid, n in field.nodes.items()}
            local_params = {
                "decay_rate": cfg.decay_rate,
                "tension_threshold": cfg.tension_threshold,
                "phase_coupling": cfg.phase_coupling,
                "bandwidth": cfg.bandwidth,
            }
            field._circuit_breakers["FederatedSync"].call(field.federated.sync_with_peers, local_phases, local_params)

        # Shard updates
        if cfg.sparse_routing and step % 100 == 0 and n_nodes > cfg.num_shards * 2:
            if getattr(cfg, "bm25_topic_shards", False):
                field._circuit_breakers["ShardUpdate"].call(field._update_shard_centers_bm25)
            else:
                field._circuit_breakers["ShardUpdate"].call(field._update_shard_centers)
            if field.rl_feedback_loop:
                field.stats["avg_rl_reward"] = field.rl_feedback_loop.get_average_reward()

        # Event-driven processing
        if field.event_scheduler and step % 10 == 0:
            processed = field.event_scheduler.process_pending(field, max_events=5)
            field.stats["events_processed"] += processed
            field.stats["event_queue_depth"] = len(field.event_scheduler._event_queue)

        # Low-rank compression
        if field.low_rank_compressor and step % cfg.compression_freq == 0:
            field._compress_field()

        # Learnable kernel step
        if field.learnable_kernel and step % 5 == 0:
            field.learnable_kernel.step()

        # Causal discovery
        causal_freq = getattr(cfg, "causal_discovery_freq", 50)
        if field.causal_engine and step % max(causal_freq, 1) == 0:
            field.causal_engine.discover_causal_structure()
            for (cause, effect), edge in field.causal_engine.causal_effects.items():
                if effect in field.nodes:
                    if cause not in field.nodes[effect].causal_parents:
                        field.nodes[effect].causal_parents.append(cause)
                    field.nodes[effect].causal_strength[cause] = edge.strength
                if cause in field.nodes:
                    field.nodes[cause].causal_effects[effect] = edge.strength
            field.stats["causal_edges"] = len(field.causal_engine.causal_effects)
            if cfg.contradiction_detection:
                field.causal_engine.detect_contradictions(cfg.contradiction_threshold)
                field.stats["contradictions"] = len(field.causal_engine.contradictions)

        # Security / integrity
        if field.security and cfg.causal_graph_integrity_check and step % 100 == 0:
            integrity = field.security.validate_causal_graph_integrity(field.causal_engine)
            if not integrity["is_valid"]:
                field.stats["security_violations"] += len(integrity["issues"])

        # Meta-memory reflection
        if field.meta_memory_eval and field.meta_memory_eval.should_reflect():
            field.meta_memory_eval.self_reflect(field)
            field.stats["meta_reflections"] += 1
            adaptive = field.meta_memory_eval.get_adaptive_params()
            if adaptive["consolidation_multiplier"] != 1.0:
                cfg.tension_threshold *= adaptive["consolidation_multiplier"]
                cfg.tension_threshold = max(0.05, min(0.5, cfg.tension_threshold))

        if field.security:
            field.stats["security_violations"] = len(field.security._violation_log)

        if field.version_control:
            field.stats["current_version"] = field.version_control.current_version
            field.stats["n_versions"] = field.version_control.n_versions

        if field.role_router and step % 5 == 0:
            field.role_router.update_kuramoto_phases(field.nodes)
            field.stats["n_shards"] = len(field.role_router.shards)
            field.stats["shard_distribution"] = {r: len(s.node_ids) for r, s in field.role_router.shards.items()}
            field.stats["role_router_enabled"] = True
            field.stats["cross_shard_exchanges"] = sum(
                s.n_cross_shard_exchanges for s in field.role_router.shards.values()
            )

        if step % 100 == 0:
            integrity = field._check_field_integrity()
            if integrity["n_issues"] > 0:
                logger.warning(f"Field integrity issues at step {step}: {integrity['n_issues']} issues")
                field.stats["field_integrity_issues"] = integrity["n_issues"]

        if step % 50 == 0:
            total = field._tension_cache_hits + field._tension_cache_misses
            field.stats["tension_cache_hits"] = field._tension_cache_hits
            field.stats["tension_cache_misses"] = field._tension_cache_misses
            field.stats["tension_cache_hit_rate"] = (field._tension_cache_hits / total) if total > 0 else 0.0
