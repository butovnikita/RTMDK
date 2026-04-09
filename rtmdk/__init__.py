"""RTMDK package."""

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
from rtmdk.config import (
    RTMDKConfig,
    ConsolidationMode,
    Backend,
    ContextFormat,
    FieldHealth,
    EvalMode,
)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
from rtmdk.nodes import (
    MemoryNode,
    CausalEdge,
    ContradictionRecord,
    CounterfactualResult,
    AgentPlan,
    ToolCall,
    Hypothesis,
    EvalResult,
    FederatedNode,
    GoalNode,
)

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
from rtmdk.utils import (
    detect_modality,
    detect_tier,
    poincare_dist,
    exp_map_poincare,
    log_map_poincare,
    mobius_add,
)
from rtmdk.utils.attention import apply_attention_bias, format_cognitive_context
from rtmdk.utils.formatting import format_context, build_system_prompt

# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------
from rtmdk.engines.predictive import PredictiveCodingModel
from rtmdk.engines.counterfactual import ScenarioPlanner
from rtmdk.engines.privacy import DifferentialPrivacy
from rtmdk.engines.causal import CausalInferenceEngine
from rtmdk.engines.neural_ode import NeuralODEDynamics

# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------
from rtmdk.support.meta_controller import MetaController
from rtmdk.support.kuramoto import KuramotoSync, FederatedRTMDK
from rtmdk.support.meta_adaptive import MetaAdaptiveKernel
from rtmdk.support.healer import TopologyHealer
from rtmdk.support.projection import IncPCAProjection
from rtmdk.support.bm25 import BM25Index
from rtmdk.support.threshold import AdaptiveThreshold
from rtmdk.support.tda import TDAMonitor
from rtmdk.support.hnsw import HNSWIndex
from rtmdk.support.torch_backend import TorchBackend
from rtmdk.support.learnable import LearnableKernel, DifferentiableConsolidation
from rtmdk.support.goal_tracker import GoalTracker
from rtmdk.support.rl_feedback import RLFeedbackLoop
from rtmdk.support.event_driven import EventDrivenScheduler, LowRankCompressor
from rtmdk.support.meta_memory import MetaMemoryEvaluator
from rtmdk.support.security import SecurityValidator
from rtmdk.support.swarm import SwarmConsensusProtocol
from rtmdk.support.agents import AgentPlanner, HypothesisVerifier, ToolRouter
from rtmdk.support.production import ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager
# Phase 15
from rtmdk.support.version_control import VersionControl, NodeDelta, Version, DiffResult
from rtmdk.support.entropy_controller import EntropyController
from rtmdk.support.triton_backend import TritonBackend, TRITON_AVAILABLE
# Phase 16
from rtmdk.support.symbolic_overlay import SymbolicOverlay, SymbolicRule, SymbolicInference
from rtmdk.support.safety_certifier import SafetyCertifier, LyapunovFunction
from rtmdk.support.ump import UniversalMemoryProtocol, UMP_VERSION, UMP_SCHEMA
# Phase 17
from rtmdk.support.role_shard_router import RoleShardRouter, RoleShard, RoleDetector, DEFAULT_ROLE

# Phase 18: Engrams
from rtmdk.engrams import EngramPattern, EngramIndex, EngramManager, PatternCompleter

# Phase 19: Advanced
from rtmdk.production.offline_dreamer import OfflineDreamer
from rtmdk.engines.causal_traversal import CausalTraversalEngine, CausalExplanationGenerator
from rtmdk.engines.ssm_dynamics import SSMDynamics
from rtmdk.engines.trust_consensus import TrustConsensusEngine, TrustDAG
from rtmdk.engines.neuro_symbolic_prover import NeuroSymbolicProver

# Phase 19 (Research modes)
from rtmdk.production.active_inference import ActiveInferenceLoop
from rtmdk.production.tpr import TensorProductRepresentation
from rtmdk.production.adversarial_arena import AdversarialArena

# ---------------------------------------------------------------------------
# Core classes from monolithic file (not yet modularized)
# ---------------------------------------------------------------------------
import sys
import os
_monolith_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _monolith_dir not in sys.path:
    sys.path.insert(0, _monolith_dir)
from rtmdk_memory_v8 import RTMDKField, RTMDKMemory

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Config
    "RTMDKConfig",
    "ConsolidationMode",
    "Backend",
    "ContextFormat",
    "FieldHealth",
    "EvalMode",
    # Nodes
    "MemoryNode",
    "CausalEdge",
    "ContradictionRecord",
    "CounterfactualResult",
    "AgentPlan",
    "ToolCall",
    "Hypothesis",
    "EvalResult",
    "FederatedNode",
    "GoalNode",
    # Utils
    "detect_modality",
    "detect_tier",
    "poincare_dist",
    "exp_map_poincare",
    "log_map_poincare",
    "mobius_add",
    "apply_attention_bias",
    "format_cognitive_context",
    "format_context",
    "build_system_prompt",
    # Engines
    "PredictiveCodingModel",
    "ScenarioPlanner",
    "DifferentialPrivacy",
    "CausalInferenceEngine",
    "NeuralODEDynamics",
    # Support
    "MetaController",
    "KuramotoSync",
    "FederatedRTMDK",
    "MetaAdaptiveKernel",
    "TopologyHealer",
    "IncPCAProjection",
    "BM25Index",
    "AdaptiveThreshold",
    "TDAMonitor",
    "HNSWIndex",
    "TorchBackend",
    "LearnableKernel",
    "DifferentiableConsolidation",
    "GoalTracker",
    "RLFeedbackLoop",
    "LowRankCompressor",
    "EventDrivenScheduler",
    "MetaMemoryEvaluator",
    "SecurityValidator",
    "SwarmConsensusProtocol",
    "AgentPlanner",
    "HypothesisVerifier",
    "ToolRouter",
    "ShadowModeEvaluator",
    "RAGASPlusEvaluator",
    "AutoRollbackManager",
    # Phase 18: Engrams
    "EngramPattern", "EngramIndex", "EngramManager", "PatternCompleter",
    # Phase 15
    "VersionControl", "NodeDelta", "Version", "DiffResult",
    "EntropyController",
    "TritonBackend", "TRITON_AVAILABLE",
    # Phase 16
    "SymbolicOverlay", "SymbolicRule", "SymbolicInference",
    "SafetyCertifier", "LyapunovFunction",
    "UniversalMemoryProtocol", "UMP_VERSION", "UMP_SCHEMA",
    # Phase 17
    "RoleShardRouter", "RoleShard", "RoleDetector", "DEFAULT_ROLE",
    # Phase 18: Engrams
    "EngramPattern", "EngramIndex", "EngramManager", "PatternCompleter",
    # Phase 19: Advanced
    "OfflineDreamer", "DreamTask",
    "CausalTraversalEngine", "CausalExplanationGenerator",
    "SSMDynamics",
    "TrustConsensusEngine", "TrustDAG",
    "NeuroSymbolicProver",
    # Phase 19 (Research)
    "ActiveInferenceLoop", "TensorProductRepresentation", "AdversarialArena",
    # Monolithic core
    "RTMDKField",
    "RTMDKMemory",
]
