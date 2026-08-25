"""SecurityInitializer — R10.1 split from FieldInitializer.

Handles lazy engines, agent orchestration, production, federated,
predictive, goal/RL, engrams, version/role, managers.
See field_initializer.py:92 and docs/RISKS.md R10.1.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtmdk.memory.field import RTMDKField

from rtmdk.memory.config import RTMDKConfig
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


class SecurityInitializer:
    """Wires production/security/federated subsystems."""

    def __init__(self, field: "RTMDKField", config: RTMDKConfig, projection_matrix=None, wal_path=None):
        self.field = field
        self.cfg = config
        self.projection_matrix = projection_matrix
        self.wal_path = wal_path

    def initialize(self) -> None:
        self._init_lazy_engines()
        self._init_agent_orchestration()
        self._init_production_mode()
        self._init_federated()
        self._init_predictive_counterfactual_dp()
        self._init_goal_rl_event_lowrank()
        self._init_engram()
        self._init_meta_memory_security()
        self._init_version_and_role()
        self._init_managers()

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
        from collections import deque

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
