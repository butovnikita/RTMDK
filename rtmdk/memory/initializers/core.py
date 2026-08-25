"""CoreInitializer — R10.1 split from FieldInitializer.

Handles core field state, caches, projection, adaptive, lifecycle.
See field_initializer.py:92 and docs/RISKS.md R10.1.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField

from rtmdk.memory.config import Backend, RTMDKConfig
from rtmdk.memory.quantization import QuantizationHelper
from rtmdk.memory.resonance import ResonanceEngine
from rtmdk.memory.cache_manager import NodeCacheManager
from rtmdk.memory.conformal import ConformalCalibrator
from rtmdk.memory.kalman import KalmanFilter
from rtmdk.support.circuit_breaker import CircuitBreaker
from rtmdk.support.threshold import AdaptiveThreshold
from rtmdk.support.tda import TDAMonitor
from rtmdk.support.torch_backend import TorchBackend
from rtmdk.support.learnable import LearnableKernel, DifferentiableConsolidation
from rtmdk.support.meta_adaptive import MetaAdaptiveKernel
from rtmdk.support.healer import TopologyHealer

logger = logging.getLogger(__name__)


class CoreInitializer:
    """Wires core subsystems (quant, rng, wal, caches, projection, adaptive)."""

    def __init__(self, field: "RTMDKField", config: RTMDKConfig, projection_matrix=None, wal_path=None):
        self.field = field
        self.cfg = config
        self.projection_matrix = projection_matrix
        self.wal_path = wal_path

    def initialize(self) -> None:
        f = self.field
        cfg = self.cfg
        # Core field basics (must be first — other inits depend on cfg/rng/quant/nodes)
        f.cfg = cfg
        f._quant = QuantizationHelper(cfg.quantization)
        f._rng = np.random.default_rng(cfg.seed)
        f.nodes = {}
        f.node_index = []
        f._resonance_engine = ResonanceEngine(
            cfg=cfg,
            meta_kernel=None,
            learnable_kernel=None,
            causal_engine=None,
            gpu_backend=None,
            quant=f._quant,
        )
        self._normalize_identity_projection()
        self._init_wal()
        f._dirty = False
        self._init_caches()
        self._init_conformal_and_learned()
        self._init_adaptive_bandwidth()
        self._init_adaptive_phase_coupling()
        self._init_kalman()
        self._init_projection_manager()
        self._init_adaptive_tda_gpu()
        self._init_learnable_and_diff()
        self._init_meta_and_healer()
        self._init_lifecycle_controls()
        self._init_circuit_breakers()
        self._init_tension_cache()
        self._init_deques_and_counters()
        self._init_stats()

    def _normalize_identity_projection(self) -> None:
        cfg = self.cfg
        if cfg.projection_mode == "identity":
            if cfg.latent_dim != cfg.embedding_dim:
                logger.warning(
                    f"projection_mode='identity' but latent_dim ({cfg.latent_dim}) != "
                    f"embedding_dim ({cfg.embedding_dim}). Aligning latent_dim to embedding_dim."
                )
                cfg.latent_dim = cfg.embedding_dim
                cfg.pca_n_components = cfg.latent_dim
        elif cfg.projection_mode == "pca":
            cfg.learn_projection = True

    def _init_wal(self) -> None:
        from rtmdk.memory.wal import WAL

        self.field.wal = WAL(
            self.wal_path if self.wal_path is not None else "",
            enabled=self.wal_path is not None,
            fsync_interval_ms=self.cfg.wal_fsync_interval_ms,
            batch_size=self.cfg.wal_batch_size,
        )

    def _init_caches(self) -> None:
        f = self.field
        cfg = self.cfg
        f._cache_mgr = NodeCacheManager()
        f.query_cache = None
        if cfg.query_cache_size > 0:
            from rtmdk.production.query_cache import QueryCache

            f.query_cache = QueryCache(max_size=cfg.query_cache_size, ttl_seconds=cfg.query_cache_ttl)

    def _init_conformal_and_learned(self) -> None:
        f = self.field
        cfg = self.cfg
        f.conformal_calibrator = None
        if cfg.conformal_prediction:
            f.conformal_calibrator = ConformalCalibrator(alpha=cfg.conformal_alpha)
        f.learned_consolidator = None
        if getattr(cfg, "learned_consolidation", False):
            from rtmdk.memory.learned_consolidation import LearnedConsolidator

            f.learned_consolidator = LearnedConsolidator(latent_dim=cfg.latent_dim)

    def _init_adaptive_bandwidth(self) -> None:
        f = self.field
        cfg = self.cfg
        f.adaptive_bw = None
        if getattr(cfg, "adaptive_bandwidth", False):
            from rtmdk.support.adaptive_bandwidth import AdaptiveBandwidthOptimizer

            f.adaptive_bw = AdaptiveBandwidthOptimizer(latent_dim=cfg.latent_dim)

    def _init_adaptive_phase_coupling(self) -> None:
        f = self.field
        cfg = self.cfg
        f._adaptive_pc_value = None
        f._adaptive_pc_estimated = False
        if getattr(cfg, "adaptive_phase_coupling", False):
            from rtmdk.memory.adaptive_pc import estimate_optimal_pc

            f._estimate_optimal_pc_fn = estimate_optimal_pc
        else:
            f._estimate_optimal_pc_fn = None

    def _init_kalman(self) -> None:
        f = self.field
        cfg = self.cfg
        f.kalman_filter = None
        if cfg.enable_kalman_filter:
            f.kalman_filter = KalmanFilter(
                latent_dim=cfg.latent_dim,
                process_noise=cfg.kalman_process_noise,
                measurement_noise=cfg.kalman_measurement_noise,
                init_variance=cfg.kalman_init_variance,
                diagonal_approx=cfg.kalman_diagonal_approx,
                hyperbolic=cfg.hyperbolic,
                ball_radius=cfg.ball_radius,
            )

    def _init_projection_manager(self) -> None:
        from rtmdk.memory.projection_manager import ProjectionManager

        self.field._projection_mgr = ProjectionManager(
            self.cfg, projection_matrix=self.projection_matrix, rng=self.field._rng
        )

    def _init_adaptive_tda_gpu(self) -> None:
        f = self.field
        cfg = self.cfg
        f.adaptive_threshold = (
            AdaptiveThreshold(cfg.adaptive_window, cfg.tension_threshold) if cfg.adaptive_threshold else None
        )
        f.tda_monitor = TDAMonitor() if cfg.tda_monitoring else None
        f.gpu_backend = TorchBackend() if cfg.backend == Backend.TORCH else None
        if f.gpu_backend and not f.gpu_backend.available:
            f.gpu_backend = None
        f._resonance_engine.gpu_backend = f.gpu_backend

    def _init_learnable_and_diff(self) -> None:
        f = self.field
        cfg = self.cfg
        f.learnable_kernel = None
        f.diff_consolidation = None
        if cfg.differentiable:
            f.learnable_kernel = LearnableKernel(cfg.bandwidth, cfg.phase_coupling, cfg.decay_rate, cfg.gradient_clip)
            f.diff_consolidation = DifferentiableConsolidation(cfg.consolidation_loss_weight)
            f._resonance_engine.learnable_kernel = f.learnable_kernel

    def _init_meta_and_healer(self) -> None:
        f = self.field
        cfg = self.cfg
        f.meta_kernel = None
        if cfg.meta_adaptive:
            f.meta_kernel = MetaAdaptiveKernel(
                cfg.bandwidth,
                cfg.phase_coupling,
                cfg.meta_adaptation_lr,
                cfg.kurtosis_target_min,
                cfg.kurtosis_target_max,
            )
            f._resonance_engine.meta_kernel = f.meta_kernel
        f.healer = None
        if cfg.self_healing:
            f.healer = TopologyHealer(
                cfg.dead_zone_threshold,
                cfg.hyperconvergence_threshold,
                cfg.fragmentation_threshold,
                cfg.healing_strength,
                cfg.max_healing_nodes_per_step,
            )

    def _init_lifecycle_controls(self) -> None:
        f = self.field
        f._workers = []
        # R4 R10.1: inner RLock, ordering distributed -> write
        f._write_lock = threading.RLock()
        f._consolidation_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtmdk_consolidate")
        f._consolidation_future = None
        f._backpressure_events = 0
        f._heavy_modules_degraded = False
        f._last_successful_step = time.time()

    def _init_circuit_breakers(self) -> None:
        f = self.field
        f._circuit_breakers = {
            "ODEEvolve": CircuitBreaker("ODEEvolve", default=0),
            "Consolidate": CircuitBreaker("Consolidate", default=[]),
            "SelfHeal": CircuitBreaker("SelfHeal"),
            "PredictorFreeEnergy": CircuitBreaker("PredictorFreeEnergy", default=0.0),
            "PredictorUpdate": CircuitBreaker("PredictorUpdate"),
            "SelfSupervise": CircuitBreaker("SelfSupervise"),
            "TDA": CircuitBreaker("TDA"),
            "MetaKernelAdapt": CircuitBreaker("MetaKernelAdapt"),
            "MetaControllerOptimize": CircuitBreaker("MetaControllerOptimize", default={}),
            "MetaControllerApply": CircuitBreaker("MetaControllerApply"),
            "FederatedSync": CircuitBreaker("FederatedSync"),
            "ODESmoothness": CircuitBreaker("ODESmoothness", default=1.0),
            "ShardUpdate": CircuitBreaker("ShardUpdate"),
        }

    def _init_tension_cache(self) -> None:
        f = self.field
        f._tension_cache = {}
        f._tension_cache_max_age = 25
        f._tension_cache_hits = 0
        f._tension_cache_misses = 0

    def _init_deques_and_counters(self) -> None:
        from collections import deque

        f = self.field
        cfg = self.cfg
        f._step_counter = 0
        f._add_node_timestamps = deque(maxlen=1000)
        f._rollback_history = deque(maxlen=cfg.max_rollback_history)
        f._stability_buffer = deque(maxlen=cfg.field_stability_window)
        f._active_node_history = deque(maxlen=50)
        f._semantic_phase_cache = {}

    def _init_stats(self) -> None:
        f = self.field
        cfg = self.cfg
        f.stats = {
            "total_adds": 0,
            "total_queries": 0,
            "consolidations": 0,
            "avg_response": 0.0,
            "active_nodes": 0,
            "projection_updates": 0,
            "self_sup_checks": 0,
            "tda_checks": 0,
            "bm25_fallbacks": 0,
            "adaptive_threshold_value": cfg.tension_threshold,
            "false_merges": 0,
            "field_stability": 1.0,
            "causal_edges": 0,
            "contradictions": 0,
            "counterfactual_queries": 0,
            "consolidation_validations": 0,
            "blocked_consolidations": 0,
            "meta_kurtosis": 3.0,
            "meta_bandwidth": cfg.bandwidth,
            "meta_phase_coupling": cfg.phase_coupling,
            "field_health": "stable",
            "healing_events": 0,
            "healing_history": [],
            "ode_steps": 0,
            "response_smoothness": 1.0,
            "plans_created": 0,
            "hypotheses_verified": 0,
            "tool_calls": 0,
            "tool_misuse_rate": 0.0,
            "evaluations": 0,
            "shadow_comparisons": 0,
            "rollbacks": 0,
            "ragas_overall": 0.0,
            "cross_modal_queries": 0,
            "cross_modal_recall": 0.0,
            "meta_optimizations": 0,
            "meta_best_params": {},
            "federated_syncs": 0,
            "federated_order_parameter": 0.0,
            "tier_distribution": {},
            "tier_coherence": 0.0,
            "hyperbolic_enabled": cfg.hyperbolic,
            "avg_hyperbolic_dist": 0.0,
            "free_energy": 0.0,
            "prediction_error": 0.0,
            "surprise_level": 0.0,
            "scenarios_generated": 0,
            "avg_scenario_confidence": 0.0,
            "privacy_budget_spent": 0.0,
            "noise_std": 0.0,
            "updates_clipped": 0,
            "shard_hits": 0,
            "shard_misses": 0,
            "avg_shard_query_time_ms": 0.0,
            "context_tokens_saved": 0,
            "cognitive_compressions": 0,
            "crystallizations": 0,
            "crystallized_clusters": 0,
            "async_queue_depth": 0,
            "async_backpressure_events": 0,
            "active_goals": 0,
            "completed_goals": 0,
            "avg_rl_reward": 0.5,
            "reward_trend": 0.0,
            "attention_bias_applied": 0,
            "compression_ratio": 1.0,
            "compression_updates": 0,
            "events_processed": 0,
            "event_queue_depth": 0,
            "recall_accuracy": 1.0,
            "meta_reflections": 0,
            "security_violations": 0,
            "tension_spikes_blocked": 0,
            "current_version": 0,
            "n_versions": 0,
            "clarifications_generated": 0,
            "n_shards": 0,
            "shard_distribution": {},
            "cross_shard_exchanges": 0,
            "role_router_enabled": False,
            "engram_retrievals": 0,
            "engrams_created": 0,
            "engrams_merged": 0,
            "field_integrity_issues": 0,
            "backpressure_degraded_mode": 0,
            "last_backpressure_recovery": 0.0,
            "startup_warnings": [],
            "tension_cache_hits": 0,
            "tension_cache_misses": 0,
            "tension_cache_hit_rate": 0.0,
            "conformal_threshold": 0.0,
            "conformal_confidence": 0.0,
            "conformal_prediction_set_size": 0,
        }
