"""FieldInitializer — encapsulates the ~460-line RTMDKField.__init__ monster."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField
from numpy.typing import NDArray

from rtmdk.memory.config import Backend, RTMDKConfig
from rtmdk.memory.quantization import QuantizationHelper
from rtmdk.memory.resonance import ResonanceEngine
from rtmdk.memory.projection_manager import ProjectionManager
from rtmdk.memory.index_manager import IndexManager
from rtmdk.memory.cache_manager import NodeCacheManager
from rtmdk.memory.consolidation_manager import ConsolidationManager
from rtmdk.memory.query_manager import QueryManager
from rtmdk.memory.routing_manager import RoutingManager
from rtmdk.memory.scheduler import StepScheduler
from rtmdk.memory.topology_manager import TopologyManager
from rtmdk.memory.async_pipeline_manager import AsyncPipelineManager
from rtmdk.memory.crystallization_manager import CrystallizationManager
from rtmdk.memory.node_manager import NodeManager
from rtmdk.memory.cognitive_manager import CognitiveManager
from rtmdk.memory.operational_manager import OperationalManager
from rtmdk.memory.merge_manager import MergeManager
from rtmdk.support.circuit_breaker import CircuitBreaker
from rtmdk.support.threshold import AdaptiveThreshold
from rtmdk.support.tda import TDAMonitor
from rtmdk.support.torch_backend import TorchBackend
from rtmdk.support.learnable import LearnableKernel, DifferentiableConsolidation
from rtmdk.support.meta_adaptive import MetaAdaptiveKernel
from rtmdk.support.healer import TopologyHealer
from rtmdk.support.agents import AgentPlanner, HypothesisVerifier, ToolRouter
from rtmdk.support.production import ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager
from rtmdk.support.kuramoto import FederatedRTMDK
from rtmdk.support.goal_tracker import GoalTracker
from rtmdk.support.rl_feedback import RLFeedbackLoop
from rtmdk.support.event_driven import LowRankCompressor, EventDrivenScheduler
from rtmdk.support.meta_memory import MetaMemoryEvaluator
from rtmdk.support.security import SecurityValidator
from rtmdk.engines.privacy import DifferentialPrivacy
from rtmdk.engines.predictive import PredictiveCodingModel
from rtmdk.engines.counterfactual import ScenarioPlanner
from rtmdk.memory.kalman import KalmanFilter
from rtmdk.memory.conformal import ConformalCalibrator

logger = logging.getLogger(__name__)

try:
    from rtmdk.support.version_control import VersionControl

    VC_AVAILABLE = True
except ImportError:
    VC_AVAILABLE = False

try:
    from rtmdk.support.role_shard_router import RoleShardRouter

    ROLE_SHARD_AVAILABLE = True
except ImportError:
    ROLE_SHARD_AVAILABLE = False


class FieldInitializer:
    """Responsible for wiring every subsystem into an RTMDKField instance."""

    def __init__(
        self,
        field: "RTMDKField",
        config: RTMDKConfig,
        projection_matrix: Optional[NDArray] = None,
        wal_path: Optional[str] = None,
    ) -> None:
        self.field = field
        self.cfg = config
        self.projection_matrix = projection_matrix
        self.wal_path = wal_path

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Wire all subsystems into the RTMDKField instance.

        Creates engines, indexes, managers, security, schedulers,
        and every other subsystem declared in the configuration.
        This is the single constructor-injection point for the field.
        """
        f = self.field
        cfg = self.cfg

        f.cfg = cfg
        f._quant = QuantizationHelper(cfg.quantization)
        f._rng = np.random.default_rng(cfg.seed)
        f.nodes = {}
        f.node_index = []

        # Early: ResonanceEngine (other subsystems register against it)
        f._resonance_engine = ResonanceEngine(
            cfg=cfg,
            meta_kernel=None,
            learnable_kernel=None,
            causal_engine=None,
            gpu_backend=None,
            quant=f._quant,
        )

        self._normalize_identity_projection()
        self._init_tiered_storage()
        self._init_wal()
        f._dirty = False
        self._init_caches()
        self._init_conformal_and_learned()
        self._init_adaptive_bandwidth()
        self._init_adaptive_phase_coupling()
        self._init_kalman()
        self._init_projection_manager()
        self._init_adaptive_tda_gpu()
        self._init_index_manager()
        f._batch_resonance_fn = None  # populated by QueryManager later
        self._init_learnable_and_diff()
        self._init_meta_and_healer()
        self._init_lazy_engines()
        self._init_agent_orchestration()
        self._init_production_mode()
        self._init_federated()
        self._init_predictive_counterfactual_dp()
        self._init_sparse_routing()
        self._init_crystallization_counters()
        self._init_lifecycle_controls()
        self._init_circuit_breakers()
        self._init_tension_cache()
        self._init_async_pipeline()
        self._init_goal_rl_event_lowrank()
        self._init_engram()
        self._init_meta_memory_security()
        self._init_version_and_role()
        self._init_stats()
        self._init_deques_and_counters()
        self._init_managers()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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

    def _init_tiered_storage(self) -> None:
        f = self.field
        cfg = self.cfg
        f._tiered_store = None
        if cfg.tiered_storage_v2_enabled:
            from rtmdk.storage.tiered import TieredNodeStore
            from rtmdk.storage.tiered_adapter import TieredNodeStoreAdapter

            hot_limit = max(1, int(cfg.max_nodes * cfg.tiered_hot_pct)) if cfg.max_nodes else 100
            warm_limit = max(1, int(cfg.max_nodes * cfg.tiered_warm_pct)) if cfg.max_nodes else 1000
            cold_dir = cfg.tiered_storage_path or "./rtmdk_cold_storage_v2"
            inner = TieredNodeStore(
                max_hot=hot_limit, max_warm=warm_limit, cold_dir=cold_dir, latent_dim=cfg.latent_dim
            )
            f._tiered_store = TieredNodeStoreAdapter(inner)
            f.nodes = f._tiered_store  # type: ignore[assignment]
        elif cfg.tiered_storage_enabled:
            from rtmdk.memory.tiered_storage import TieredNodeStore as LegacyTieredNodeStore

            hot_limit = max(1, int(cfg.max_nodes * cfg.tiered_hot_pct)) if cfg.max_nodes else 100
            warm_limit = max(1, int(cfg.max_nodes * cfg.tiered_warm_pct)) if cfg.max_nodes else 1000
            cold_dir = cfg.tiered_storage_path or "./rtmdk_cold_storage"
            f._tiered_store = LegacyTieredNodeStore(hot_limit, warm_limit, cold_dir, cfg.latent_dim)
            f.nodes = f._tiered_store  # type: ignore[assignment]

    def _init_wal(self) -> None:
        from rtmdk.memory.wal import WAL

        self.field.wal = WAL(
            # Path is unused when the WAL is disabled (all I/O is guarded by `enabled`)
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

    def _init_index_manager(self) -> None:
        f = self.field
        cfg = self.cfg
        f._index_mgr = IndexManager(cfg, cfg.latent_dim, f._rng, f._quant)
        f.bm25_index = f._index_mgr.bm25_index
        f.hnsw_index = f._index_mgr.hnsw_index
        f.shard_centers = f._index_mgr.shard_centers
        f._async_index_builder = f._index_mgr._async_builder

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

    def _init_lazy_engines(self) -> None:
        f = self.field
        cfg = self.cfg
        f._causal_engine = None
        f._causal_engine_initialized = cfg.causal_topological
        f._meta_controller = None
        f._meta_controller_initialized = cfg.meta_controller

    def _init_agent_orchestration(self) -> None:
        f = self.field
        cfg = self.cfg
        f.agent_planner = None
        f.hypothesis_verifier = None
        f.tool_router = None
        if cfg.agent_orchestration:
            f.agent_planner = AgentPlanner(cfg.max_plan_depth, cfg.max_tool_calls, cfg.tool_timeout)
            f.hypothesis_verifier = HypothesisVerifier(cfg.verification_confidence_threshold)
            f.tool_router = ToolRouter(cfg.tool_timeout)

    def _init_production_mode(self) -> None:
        f = self.field
        cfg = self.cfg
        f.shadow_evaluator = None
        f.ragas_evaluator = None
        f.rollback_manager = None
        if cfg.production_mode:
            if cfg.shadow_mode:
                f.shadow_evaluator = ShadowModeEvaluator(cfg.shadow_fallback_threshold)
            if cfg.ragas_enabled:
                f.ragas_evaluator = RAGASPlusEvaluator()
            if cfg.auto_rollback:
                f.rollback_manager = AutoRollbackManager(cfg.auto_rollback_threshold)

    def _init_federated(self) -> None:
        f = self.field
        cfg = self.cfg
        f.federated = None
        if cfg.federated:
            f.federated = FederatedRTMDK(
                node_id=cfg.node_id,
                sync_lr=cfg.federated_sync_lr,
                sync_freq=cfg.federated_sync_freq,
                min_resonance=cfg.federated_min_resonance,
            )

    def _init_predictive_counterfactual_dp(self) -> None:
        f = self.field
        cfg = self.cfg
        f.predictor = None
        if cfg.predictive_coding:
            f.predictor = PredictiveCodingModel(cfg.latent_dim, lr=cfg.pc_lr)
        f._state_history = deque(maxlen=100)

        f.scenario_planner = None
        if cfg.counterfactual_imagination:
            f.scenario_planner = ScenarioPlanner(f, max_scenarios=cfg.max_scenarios)

        f.dp = None
        if cfg.differential_privacy:
            f.dp = DifferentialPrivacy(cfg.dp_epsilon, cfg.dp_delta, cfg.dp_max_norm)

    def _init_sparse_routing(self) -> None:
        f = self.field
        cfg = self.cfg
        f.shard_router = None
        f._node_shard_map = {}
        if cfg.sparse_routing:
            f.shard_router = np.zeros(cfg.num_shards, dtype=np.float32)

    def _init_crystallization_counters(self) -> None:
        f = self.field
        f._crystallization_counter = 0
        f._crystallized_nodes = set()

    def _init_lifecycle_controls(self) -> None:
        f = self.field
        f._workers = []
        # R4 (2026-08-24): _write_lock is the inner intra-process RLock.
        # Lock ordering (R4.3): distributed_lock (outer, file/redis, core.py:retrieve_nodes)
        # -> _write_lock (inner, query_manager snapshot, SOT update, add_node).
        # Never acquire in reverse order. RLock allows re-entrancy for nested query paths.
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

    def _init_async_pipeline(self) -> None:
        f = self.field
        cfg = self.cfg
        f.query_q = None
        f.save_q = None
        f.evolve_q = None
        f._workers_started = False
        if cfg.async_pipeline:
            f.query_q = asyncio.Queue(maxsize=cfg.query_queue_size)
            f.save_q = asyncio.Queue(maxsize=cfg.save_queue_size)
            f.evolve_q = asyncio.Queue(maxsize=cfg.evolve_queue_size)

    def _init_goal_rl_event_lowrank(self) -> None:
        f = self.field
        cfg = self.cfg
        f.goal_tracker = None
        if cfg.goal_tracking:
            f.goal_tracker = GoalTracker(cfg.max_goals, cfg.goal_decay, cfg.goal_completion_threshold)

        f.rl_feedback_loop = None
        if cfg.rl_feedback:
            f.rl_feedback_loop = RLFeedbackLoop(cfg.rl_learning_rate, cfg.rl_reward_window)

        f.event_scheduler = None
        f.low_rank_compressor = None
        if cfg.event_driven:
            f.event_scheduler = EventDrivenScheduler()
        if cfg.low_rank_compression:
            f.low_rank_compressor = LowRankCompressor(cfg.compression_rank)

    def _init_engram(self) -> None:
        f = self.field
        cfg = self.cfg
        f.engram_manager = None
        if cfg.enable_engrams:
            try:
                from rtmdk.engrams import EngramManager

                f.engram_manager = EngramManager(
                    min_nodes=cfg.engram_min_nodes,
                    max_nodes=cfg.engram_max_nodes,
                    creation_threshold=cfg.engram_creation_threshold,
                    decay_rate=cfg.engram_decay_rate,
                    pattern_completion=cfg.engram_pattern_completion,
                    overlap_threshold=cfg.engram_overlap_threshold,
                )
            except Exception:
                logger.warning("Engram manager initialization failed in RTMDKField, disabling", exc_info=True)
                f.engram_manager = None

    def _init_meta_memory_security(self) -> None:
        f = self.field
        cfg = self.cfg
        f.meta_memory_eval = None
        if cfg.meta_memory:
            f.meta_memory_eval = MetaMemoryEvaluator(
                cfg.recall_accuracy_threshold, cfg.memory_age_factor, cfg.self_reflection_freq
            )

        f.security = None
        if cfg.security_enabled:
            f.security = SecurityValidator(
                cfg.max_node_text_length, cfg.tension_spike_threshold, cfg.prompt_injection_patterns
            )

    def _init_version_and_role(self) -> None:
        f = self.field
        cfg = self.cfg
        f.version_control = None
        if cfg.version_control and VC_AVAILABLE:
            f.version_control = VersionControl(max_versions=cfg.max_versions)
        elif cfg.version_control and not VC_AVAILABLE:
            logger.error("version_control enabled but rtmdk.support.version_control not available — feature disabled")
            f.stats.setdefault("startup_warnings", []).append("version_control unavailable")

        f.role_router = None
        if cfg.role_sharding and ROLE_SHARD_AVAILABLE:
            f.role_router = RoleShardRouter(
                shards=cfg.role_shards,
                cross_shard_threshold=cfg.cross_shard_threshold,
                auto_role_detection=cfg.auto_role_detection,
            )
        elif cfg.role_sharding and not ROLE_SHARD_AVAILABLE:
            logger.error("role_sharding enabled but rtmdk.support.role_shard_router not available — feature disabled")
            f.stats.setdefault("startup_warnings", []).append("role_shard_router unavailable")

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

    def _init_deques_and_counters(self) -> None:
        f = self.field
        cfg = self.cfg
        f._step_counter = 0
        f._add_node_timestamps = deque(maxlen=1000)
        f._rollback_history = deque(maxlen=cfg.max_rollback_history)
        f._stability_buffer = deque(maxlen=cfg.field_stability_window)
        f._active_node_history = deque(maxlen=50)
        f._semantic_phase_cache = {}

    def _init_managers(self) -> None:
        f = self.field
        f._consolidation_mgr = ConsolidationManager(f)
        f._query_mgr = QueryManager(f)
        f._routing_mgr = RoutingManager(f)
        f._topology_mgr = TopologyManager(f)
        f._async_pipeline_mgr = AsyncPipelineManager(f)
        f._crystallization_mgr = CrystallizationManager(f)
        f._node_mgr = NodeManager(f)
        f._cognitive_mgr = CognitiveManager(f)
        f._operational_mgr = OperationalManager(f)
        f._merge_mgr = MergeManager(f)
        f._scheduler = StepScheduler(f)
