"""RTMDK support modules."""
from __future__ import annotations

from .meta_controller import MetaController
from .kuramoto import KuramotoSync, FederatedRTMDK
from .meta_adaptive import MetaAdaptiveKernel
from .healer import TopologyHealer
from .projection import IncPCAProjection
from .bm25 import BM25Index
from .threshold import AdaptiveThreshold
from .tda import TDAMonitor
from .hnsw import HNSWIndex
from .torch_backend import TorchBackend
from .learnable import LearnableKernel, DifferentiableConsolidation
from .goal_tracker import GoalTracker
from .rl_feedback import RLFeedbackLoop
from .event_driven import LowRankCompressor, EventDrivenScheduler
from .meta_memory import MetaMemoryEvaluator
from .security import SecurityValidator
from .swarm import SwarmConsensusProtocol
from .agents import AgentPlanner, HypothesisVerifier, ToolRouter
from .production import ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager
# Phase 15
from .version_control import VersionControl, NodeDelta, Version, DiffResult
from .entropy_controller import EntropyController
from .triton_backend import TritonBackend, TRITON_AVAILABLE
# Phase 16
from .symbolic_overlay import SymbolicOverlay, SymbolicRule, SymbolicInference
from .safety_certifier import SafetyCertifier, LyapunovFunction
from .ump import UniversalMemoryProtocol, UMP_VERSION, UMP_SCHEMA
# Phase 17
from .role_shard_router import RoleShardRouter, RoleShard, RoleDetector, DEFAULT_ROLE

__all__ = [
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
]
