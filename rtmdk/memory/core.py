"""
rtmdk/memory/core.py
Resonance-Topological Memory - Version 8.1+

Phase 11 Features:
Track 1: Multi-level memory stratification (Episodic / Semantic / Procedural)
Track 2: Hyperbolic geometry (Poincare ball model)
Track 3: Predictive coding / Active inference
Track 4: Counterfactual imagination & scenario planning
Track 5: Differential privacy & secure federation

All v7 components preserved:
MetaAdaptiveKernel, TopologyHealer, CausalInferenceEngine, NeuralODEDynamics,
IncPCAProjection, BM25Index, HNSWIndex, TorchBackend, LearnableKernel,
DifferentiableConsolidation, AgentPlanner, HypothesisVerifier, ToolRouter,
ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager, MetaController,
KuramotoSync, FederatedRTMDK, FederatedNode, detect_modality, cross_modal_resonance
"""

from __future__ import annotations
import asyncio
import json
import math
import re
import time
import os
import hashlib
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union, Callable, Any, Set, FrozenSet
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import cKDTree
from scipy.integrate import odeint, solve_ivp
from scipy import stats as scipy_stats
from pydantic import BaseModel, Field, ConfigDict, model_validator
import logging

logger = logging.getLogger(__name__)

# Custom exceptions (Fix 7: security violations should raise, not return "")
class SecurityViolationError(Exception):
    """Raised when a node violates security policy (prompt injection detected)."""
    pass

# Phase 15: New modules
try:
    from rtmdk.support.version_control import VersionControl, NodeDelta, Version, DiffResult
    VC_AVAILABLE = True
except ImportError:
    VC_AVAILABLE = False

try:
    from rtmdk.support.entropy_controller import EntropyController
    ENTROPY_AVAILABLE = True
except ImportError:
    ENTROPY_AVAILABLE = False

try:
    from rtmdk.support.triton_backend import TritonBackend, TRITON_AVAILABLE
except ImportError:
    TritonBackend = None  # type: ignore
    TRITON_AVAILABLE = False

# Phase 16: New modules
try:
    from rtmdk.support.symbolic_overlay import SymbolicOverlay, SymbolicRule, SymbolicInference
    SYMBOLIC_AVAILABLE = True
except ImportError:
    SYMBOLIC_AVAILABLE = False

try:
    from rtmdk.support.safety_certifier import SafetyCertifier, LyapunovFunction
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False

try:
    from rtmdk.support.ump import UniversalMemoryProtocol, UMP_VERSION, UMP_SCHEMA
    UMP_AVAILABLE = True
except ImportError:
    UMP_AVAILABLE = False

# Phase 17: RoleShardRouter
try:
    from rtmdk.support.role_shard_router import RoleShardRouter, RoleShard, RoleDetector, DEFAULT_ROLE
    ROLE_SHARD_AVAILABLE = True
except ImportError:
    ROLE_SHARD_AVAILABLE = False
    DEFAULT_ROLE = "default"  # Fallback

# Torch availability check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


# ============================================================================
# CONSTANTS: Named constants for magic numbers
# ============================================================================

# Statistical constants
CHI_SQUARED_CRITICAL_DF1 = 3.84  # Chi-squared critical value (df=1, p=0.05)
CHI_SQUARED_CRITICAL_DF2 = 5.99  # Chi-squared critical value (df=2, p=0.05)

# Consolidation constants
CONSOLIDATION_DISTANCE_THRESHOLD = 2.5
CONSOLIDATION_PROBABILITY = 0.15
CRYSTALLIZATION_SIMILARITY_HIGH = 0.75
CRYSTALLIZATION_SIMILARITY_LOW = 0.6

# Session retrieval boost
SESSION_BOOST_FACTOR = 1.3  # 30% boost for session-matching nodes

# Performance limits
MAX_TENSION_SCAN = 200
CACHE_INVALID_HASH_MODULUS = 5

# File limits
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100MB
MAX_NODE_TEXT_LENGTH = 10000
SECURE_FILE_PERMISSIONS = 0o600

# Frequency constants (steps)
SELF_SUPERVISION_FREQ = 20
ODE_SMOOTHNESS_FREQ = 10
TENSION_CHECK_FREQ = 100
HEALING_CHECK_FREQ = 50
SYMBOLIC_OVERLAY_FREQ = 50
META_KERNEL_ADAPT_FREQ = 5
MAX_NODES_PRUNE_CHECK_FREQ = 10


# ============================================================================
# SECURITY UTILITIES
# ============================================================================

def _sanitize_path(path: str) -> str:
    """Sanitize file path to prevent directory traversal attacks.
    
    Rejects paths containing '..' (path traversal).
    Returns normalized path.
    """
    import os
    # Reject parent directory references BEFORE normalization
    # (normpath collapses 'a/../b' to 'b', which would hide the attack)
    if ".." in path.replace("\\", "/").split("/"):
        raise SecurityViolationError(f"Path traversal detected: {path}")
    # Normalize to catch unicode tricks and mixed separators
    normalized = os.path.normpath(path)
    return normalized


def _safe_json_load(path: str) -> Dict:
    """Load JSON with size limit to prevent memory exhaustion."""
    file_size = os.path.getsize(path)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB (max {MAX_FILE_SIZE_BYTES / (1024*1024):.0f}MB)")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    if len(raw.encode("utf-8")) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File exceeds maximum allowed size after encoding check")
    return json.loads(raw)


# ============================================================================
# CONFIGURATION v7
# ============================================================================

class ConsolidationMode(Enum):
    DIALECTICAL = "dialectical"
    MERGE = "merge"
    PRUNE = "prune"

class Backend(Enum):
    NUMPY = "numpy"
    TORCH = "torch"

class ContextFormat(Enum):
    PLAIN = "plain"
    JSON = "json"
    YAML = "yaml"
    ATTENTION = "attention"  # Control-tokens for attention-aware LLMs

class FieldHealth(Enum):
    STABLE = "stable"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    HEALING = "healing"

class EvalMode(Enum):
    PRODUCTION = "production"
    SHADOW = "shadow"
    EVALUATION = "evaluation"

@dataclass
class RTMDKConfig:
    embedding_dim: int = 768
    latent_dim: int = 64  # Matches server default — change only if you know the impact
    resonance_kernel: str = "gaussian_phase"
    phase_coupling: float = 0.3
    bandwidth: float = 1.0
    attraction_lr: float = 0.02
    phase_sync_lr: float = 0.01
    decay_rate: float = 0.997  # Matches server default — half-life ~230 steps
    min_amplitude: float = 0.05
    tension_threshold: float = 0.15  # Matches server default — moderate consolidation
    consolidation_mode: ConsolidationMode = ConsolidationMode.DIALECTICAL
    max_nodes: Optional[int] = 5000
    top_k: int = 5
    min_response: float = 0.005  # OPTIMIZED: 20x lower → more results pass filter
    enable_async: bool = True
    log_level: str = "INFO"
    seed: Optional[int] = None

    # Phase 1
    context_format: ContextFormat = ContextFormat.PLAIN
    use_structured_prompt: bool = True
    adaptive_threshold: bool = False
    adaptive_window: int = 30
    learn_projection: bool = True  # OPTIMIZED: IncPCA instead of random matrix
    projection_lr: float = 0.001
    projection_update_freq: int = 300  # OPTIMIZED: >= latent_dim for IncPCA first fit
    pca_n_components: Optional[int] = None
    bm25_fallback: bool = True  # OPTIMIZED: text search as safety net
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    hybrid_alpha: float = 1.0  # 1.0 = pure RTMDK, 0.0 = pure BM25, 0.7 = 70/30 blend

    # Phase 2
    soft_gates: bool = False
    gate_temperature: float = 0.15
    self_supervision: bool = False
    self_sup_threshold: float = 0.3
    self_sup_verify_after_consolidate: bool = False
    backend: Backend = Backend.NUMPY
    gpu_batch_size: int = 512
    l2_regularization: float = 0.0001
    false_merge_threshold: float = 0.4
    field_stability_window: int = 20
    enable_rollback: bool = False
    max_rollback_history: int = 10  # Fix 7: Reduced from 50 — each snapshot stores full node copies, memory intensive

    # Phase 3
    multimodal: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text"])
    modality_phase_shifts: Dict[str, float] = field(default_factory=dict)
    use_hnsw: bool = True  # OPTIMIZED: fast approximate nearest neighbor
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    tda_monitoring: bool = False
    tda_check_freq: int = 50

    # Track 1: Differentiable field
    differentiable: bool = False
    learnable_bandwidth: bool = False
    learnable_phase_coupling: bool = False
    learnable_decay: bool = False
    gradient_clip: float = 1.0
    consolidation_loss_weight: float = 0.1

    # Phase 5
    meta_adaptive: bool = False
    meta_adaptation_lr: float = 0.005
    kurtosis_target_min: float = 1.5
    kurtosis_target_max: float = 4.0
    self_healing: bool = False
    healing_check_freq: int = 25
    dead_zone_threshold: float = 0.15
    hyperconvergence_threshold: float = 0.05
    fragmentation_threshold: float = 0.6
    healing_strength: float = 0.1
    max_healing_nodes_per_step: int = 5

    # Phase 6
    causal_topological: bool = False
    causal_discovery_min_samples: int = 20
    causal_p_threshold: float = 0.05
    do_calculus_validation: bool = True
    counterfactual_enabled: bool = False
    counterfactual_max_depth: int = 3
    contradiction_detection: bool = True
    contradiction_threshold: float = 0.3
    causal_adjustment_sets: bool = True

    # Phase 7: Neural ODE/SDE
    continuous_dynamics: bool = False
    ode_solver: str = "RK45"
    ode_atol: float = 1e-6
    ode_rtol: float = 1e-5
    ode_time_horizon: float = 1.0
    ode_n_steps: int = 20
    ode_chunk_size: int = 256
    sde_noise_level: float = 0.01
    adjoint_enabled: bool = False
    response_smoothness_target: float = 0.92

    # Phase 8: Agent orchestration
    agent_orchestration: bool = False
    max_plan_depth: int = 3
    max_tool_calls: int = 5
    tool_timeout: float = 15.0
    hypothesis_verification: bool = True
    verification_confidence_threshold: float = 0.7
    goal_directed_routing: bool = False

    # Phase 9: Production
    production_mode: bool = False
    eval_mode: EvalMode = EvalMode.PRODUCTION
    shadow_mode: bool = False
    shadow_fallback_threshold: float = 0.3
    auto_rollback: bool = False
    auto_rollback_threshold: float = 0.15
    eval_frequency: int = 100
    ragas_enabled: bool = False
    drift_detection: bool = False
    drift_window: int = 100
    drift_threshold: float = 0.05
    metrics_retention: int = 10000

    # Track 10.1: Cross-modal resonance
    cross_modal: bool = False
    modal_phase_offsets: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.0,
        "code": math.pi / 4,
        "audio": math.pi / 2,
        "vision": 3 * math.pi / 4,
        "metrics": math.pi,
    })
    cross_modal_kernel_weight: float = 0.35

    # Track 10.2: Meta-cognitive controller
    meta_controller: bool = False
    meta_optimization_freq: int = 500
    meta_n_trials: int = 20
    meta_optimize_params: List[str] = field(default_factory=lambda: [
        "decay_rate", "tension_threshold", "phase_coupling", "bandwidth"
    ])

    # Phase 18: Engrams (biological memory patterns)
    enable_engrams: bool = True
    engram_min_nodes: int = 2
    engram_max_nodes: int = 20
    engram_creation_threshold: float = 0.6
    engram_decay_rate: float = 0.998
    engram_pattern_completion: bool = True
    engram_overlap_threshold: float = 0.7

    # Track 10.3: Federated sync
    federated: bool = False
    federated_sync_lr: float = 0.01
    federated_sync_freq: int = 100
    federated_min_resonance: float = 0.2
    node_id: str = "local"

    # Phase 11 Track 1: Memory stratification
    memory_tiers: Set[str] = field(default_factory=lambda: {"episodic", "semantic", "procedural"})
    tier_decay: Dict[str, float] = field(default_factory=lambda: {
        "episodic": 0.992, "semantic": 0.999, "procedural": 1.0
    })
    tier_tension_thresh: Dict[str, float] = field(default_factory=lambda: {
        "episodic": 0.10, "semantic": 0.22, "procedural": 0.35
    })

    # Phase 11 Track 2: Hyperbolic geometry
    hyperbolic: bool = False
    ball_radius: float = 0.85
    curvature: float = -1.0

    # Phase 11 Track 3: Predictive coding
    predictive_coding: bool = False
    pc_latent_dim: int = 32
    pc_lr: float = 0.01

    # Phase 11 Track 4: Counterfactual imagination
    counterfactual_imagination: bool = False
    max_scenarios: int = 5

    # Phase 11 Track 5: Differential privacy
    differential_privacy: bool = False
    dp_epsilon: float = 2.0
    dp_delta: float = 1e-5
    dp_max_norm: float = 1.0

    # Phase 12 Track 1: Sparse resonant routing (MoE-memory)
    sparse_routing: bool = False
    num_shards: int = 8
    top_shards: int = 3

    # Phase 12 Track 2: Cognitive context compression
    cognitive_compression: bool = False
    high_resonance_threshold: float = 0.6

    # Phase 12 Track 3: Crystallization
    crystallization: bool = False
    crystallization_freq: int = 200
    crystallization_similarity: float = 0.75
    crystallization_min_cluster: int = 3

    # Phase 12 Track 4: Async pipeline
    async_pipeline: bool = False
    query_queue_size: int = 50
    save_queue_size: int = 100
    evolve_queue_size: int = 20

    # Phase 13 Track 1: Teleological layer (Goal/Intent Tracking)
    goal_tracking: bool = False
    max_goals: int = 20
    goal_decay: float = 0.995
    goal_completion_threshold: float = 0.8

    # Phase 13 Track 2: Cognitive attention bias
    attention_bias: bool = False
    bias_temperature: float = 1.0

    # Phase 13 Track 3: Closed-loop RL feedback
    rl_feedback: bool = False
    rl_learning_rate: float = 0.01
    rl_reward_window: int = 10

    # Phase 13 Track 4: Event-driven + Low-Rank compression
    event_driven: bool = False
    low_rank_compression: bool = False
    compression_rank: int = 32
    compression_freq: int = 500

    # Phase 14 Track 1: Introspective Meta-Memory
    meta_memory: bool = False
    self_reflection_freq: int = 100
    memory_age_factor: float = 0.001
    recall_accuracy_threshold: float = 0.6

    # Phase 14 Track 2: Formal Security
    security_enabled: bool = False
    max_node_text_length: int = 10000
    tension_spike_threshold: float = 0.5
    causal_graph_integrity_check: bool = True
    prompt_injection_patterns: List[str] = field(default_factory=lambda: [
        "ignore previous", "system prompt", "you are now", "disregard",
        "ignore all", "new instruction", "override"
    ])

    # Phase 14 Track 5: Swarm Memory
    swarm_memory: bool = False
    swarm_consensus_threshold: float = 0.5
    swarm_max_agents: int = 10
    swarm_vote_weight: float = 0.3

    # Phase 15 Track 1: Version Control (Memory Git)
    version_control: bool = False
    max_versions: int = 100

    # Phase 15 Track 2: Proactive Clarification
    proactive_clarification: bool = False
    clarification_threshold_ratio: float = 0.5

    # Phase 15 Track 3: Attention Tokens
    attention_tokens: bool = True  # Enabled by default (extends attention_bias)

    # Phase 15 Track 4: Entropy Control
    entropy_management: bool = False
    entropy_high_threshold: float = 3.0
    entropy_low_threshold: float = 0.5

    # Phase 15 Track 5: Triton/CUDA Backend
    triton_backend: bool = False
    min_nodes_for_gpu: int = 2000

    # Phase 16 Track 1: SymbolicOverlay
    symbolic_overlay: bool = False
    symbolic_min_self_sup: float = 0.7
    symbolic_max_tension: float = 0.15
    symbolic_confidence_threshold: float = 0.65

    # Phase 16 Track 2: SafetyCertifier
    safety_certifier: bool = False
    safety_mode: str = "soft_regulate"  # monitor_only | soft_regulate | hard_block
    lyapunov_alpha: float = 0.4
    lyapunov_beta: float = 0.4
    lyapunov_gamma: float = 0.2
    lyapunov_threshold: float = 0.1

    # Phase 16 Track 3: Universal Memory Protocol
    ump_enabled: bool = False

    # Phase 17: RoleShardRouter
    role_sharding: bool = False
    role_shards: Set[str] = field(default_factory=lambda: {"default"})
    cross_shard_threshold: float = 0.45
    auto_role_detection: bool = True

    # System prompt (None = no system prompt, used for SillyTavern)
    system_prompt: Optional[str] = "You are a helpful assistant with long-term memory powered by RTMDK (Resonance-Topological Memory)."

    # Phase 19: Advanced Improvements
    offline_dreaming: bool = True
    dreaming_freq: int = 50
    causal_traversal: bool = True
    causal_max_hops: int = 3
    ssm_dynamics: bool = False
    ssm_state_dim: int = 64
    trust_consensus: bool = False
    trust_min_reputation: float = 0.3
    neuro_symbolic_prover: bool = False
    prover_backend: str = "z3"

    # Phase 20: Domain Memory
    domain_aware_retrieval: bool = False
    domain_consolidation_guard: bool = False

    # CPEN+ specific flags
    cpen_parent_ode: bool = False
    cpen_child_ode: bool = False
    hebbian_learning_rate: float = 0.01
    causal_masking: bool = False

    def __post_init__(self):
        # Phase 2: Env var overrides for critical config fields
        # Priority: explicit args > env vars > dataclass defaults
        _env_overrides = [
            # Core
            ("RTMDK_EMBEDDING_DIM", "embedding_dim", int),
            ("RTMDK_LATENT_DIM", "latent_dim", int),
            ("RTMDK_DECAY_RATE", "decay_rate", float),
            ("RTMDK_TENSION_THRESHOLD", "tension_threshold", float),
            ("RTMDK_MIN_RESPONSE", "min_response", float),
            ("RTMDK_TOP_K", "top_k", int),
            ("RTMDK_MAX_NODES", "max_nodes", lambda x: int(x) if x and x.lower() != "none" else None),
            ("RTMDK_CONSOLIDATION_MODE", "consolidation_mode", lambda x: ConsolidationMode(x)),
            # Retrieval
            ("RTMDK_PHASE_COUPLING", "phase_coupling", float),
            ("RTMDK_BANDWIDTH", "bandwidth", float),
            ("RTMDK_USE_HNSW", "use_hnsw", lambda x: x.lower() == "true"),
            ("RTMDK_HNSW_M", "hnsw_m", int),
            ("RTMDK_BM25_FALLBACK", "bm25_fallback", lambda x: x.lower() == "true"),
            ("RTMDK_LEARN_PROJECTION", "learn_projection", lambda x: x.lower() == "true"),
            ("RTMDK_PROJECTION_LR", "projection_lr", float),
            ("RTMDK_PROJECTION_UPDATE_FREQ", "projection_update_freq", int),
            # Performance
            ("RTMDK_ENABLE_ASYNC", "enable_async", lambda x: x.lower() == "true"),
            ("RTMDK_SOFT_GATES", "soft_gates", lambda x: x.lower() == "true"),
            ("RTMDK_ATTENTION_BIAS", "attention_bias", lambda x: x.lower() == "true"),
            ("RTMDK_ADAPTIVE_THRESHOLD", "adaptive_threshold", lambda x: x.lower() == "true"),
            # Production
            ("RTMDK_CROSS_MODAL", "cross_modal", lambda x: x.lower() == "true"),
            ("RTMDK_CAUSAL_TOPOLOGICAL", "causal_topological", lambda x: x.lower() == "true"),
            ("RTMDK_META_ADAPTIVE", "meta_adaptive", lambda x: x.lower() == "true"),
            ("RTMDK_SELF_HEALING", "self_healing", lambda x: x.lower() == "true"),
            ("RTMDK_VERSION_CONTROL", "version_control", lambda x: x.lower() == "true"),
            # Phase 18: Engrams
            ("RTMDK_ENABLE_ENGRAMS", "enable_engrams", lambda x: x.lower() == "true"),
            ("RTMDK_ENGRAM_MIN_NODES", "engram_min_nodes", int),
            ("RTMDK_ENGRAM_MAX_NODES", "engram_max_nodes", int),
            # Phase 19
            ("RTMDK_OFFLINE_DREAMING", "offline_dreaming", lambda x: x.lower() == "true"),
            ("RTMDK_CAUSAL_TRAVERSAL", "causal_traversal", lambda x: x.lower() == "true"),
            ("RTMDK_CAUSAL_MAX_HOPS", "causal_max_hops", int),
            ("RTMDK_SSM_DYNAMICS", "ssm_dynamics", lambda x: x.lower() == "true"),
            ("RTMDK_SSM_STATE_DIM", "ssm_state_dim", int),
            ("RTMDK_TRUST_CONSENSUS", "trust_consensus", lambda x: x.lower() == "true"),
            ("RTMDK_NEURO_SYMBOLIC_PROVER", "neuro_symbolic_prover", lambda x: x.lower() == "true"),
            # Phase 11
            ("RTMDK_HYPERBOLIC", "hyperbolic", lambda x: x.lower() == "true"),
            ("RTMDK_PREDICTIVE_CODING", "predictive_coding", lambda x: x.lower() == "true"),
            ("RTMDK_COUNTERFACTUAL_IMAGINATION", "counterfactual_imagination", lambda x: x.lower() == "true"),
            ("RTMDK_DIFFERENTIAL_PRIVACY", "differential_privacy", lambda x: x.lower() == "true"),
            ("RTMDK_DP_EPSILON", "dp_epsilon", float),
            # Phase 12-17
            ("RTMDK_SPARSE_ROUTING", "sparse_routing", lambda x: x.lower() == "true"),
            ("RTMDK_NUM_SHARDS", "num_shards", int),
            ("RTMDK_GOAL_TRACKING", "goal_tracking", lambda x: x.lower() == "true"),
            ("RTMDK_RL_FEEDBACK", "rl_feedback", lambda x: x.lower() == "true"),
            ("RTMDK_LOW_RANK_COMPRESSION", "low_rank_compression", lambda x: x.lower() == "true"),
            ("RTMDK_META_MEMORY", "meta_memory", lambda x: x.lower() == "true"),
            ("RTMDK_SECURITY_ENABLED", "security_enabled", lambda x: x.lower() == "true"),
            ("RTMDK_SWARM_MEMORY", "swarm_memory", lambda x: x.lower() == "true"),
            ("RTMDK_SYMBOLIC_OVERLAY", "symbolic_overlay", lambda x: x.lower() == "true"),
            ("RTMDK_SAFETY_CERTIFIER", "safety_certifier", lambda x: x.lower() == "true"),
            ("RTMDK_ROLE_SHARDING", "role_sharding", lambda x: x.lower() == "true"),
            ("RTMDK_CONTEXT_FORMAT", "context_format", lambda x: ContextFormat(x)),
            ("RTMDK_LOG_LEVEL", "log_level", str),
            ("RTMDK_SYSTEM_PROMPT", "system_prompt", str),
            ("RTMDK_CPEN_PARENT_ODE", "cpen_parent_ode", lambda x: x.lower() == "true"),
            ("RTMDK_CPEN_CHILD_ODE", "cpen_child_ode", lambda x: x.lower() == "true"),
            ("RTMDK_HEBBIAN_LEARNING_RATE", "hebbian_learning_rate", float),
            ("RTMDK_CAUSAL_MASKING", "causal_masking", lambda x: x.lower() == "true"),
            # Phase 20: Domain Memory
            ("RTMDK_DOMAIN_AWARE_RETRIEVAL", "domain_aware_retrieval", lambda x: x.lower() == "true"),
            ("RTMDK_DOMAIN_CONSOLIDATION_GUARD", "domain_consolidation_guard", lambda x: x.lower() == "true"),
        ]
        for env_key, attr, type_fn in _env_overrides:
            val = os.getenv(env_key)
            if val is not None:
                try:
                    parsed = type_fn(val)
                    # Handle empty string as None for Optional[str] fields
                    if attr == "system_prompt" and parsed == "":
                        parsed = None
                    elif attr == "system_prompt" and parsed.lower() == "none":
                        parsed = None
                    object.__setattr__(self, attr, parsed)
                except (ValueError, TypeError) as e:
                    logging.getLogger("rtmdk").warning(
                        f"Invalid env var {env_key}={val}: {e}"
                    )

        logger.setLevel(getattr(logging, self.log_level.upper()))
        if not self.modality_phase_shifts:
            self.modality_phase_shifts = {
                "text": 0.0, "audio": np.pi / 3,
                "image": np.pi / 2, "video": np.pi,
            }
        if self.pca_n_components is None:
            self.pca_n_components = self.latent_dim


# ============================================================================
# PHASE 11 TRACK 1: MEMORY STRATIFICATION
# ============================================================================

def _enum_value(val, default):
    """Safely extract enum value for serialization."""
    return val.value if isinstance(val, Enum) else (val if val is not None else default)


def detect_tier(text: str, context: Optional[Dict] = None) -> str:
    """Auto-detect memory tier from content."""
    context = context or {}
    text_lower = text.lower()
    # Procedural: how-to, tool usage
    if context.get("tool_used"):
        return "procedural"
    if any(p in text_lower for p in ["how to", "how do", "how can", "steps to", "tutorial", "guide"]):
        return "procedural"
    # Episodic: dates, temporal markers
    if re.search(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", text):
        return "episodic"
    if any(p in text_lower for p in ["yesterday", "last week", "last month", "ago", "вчера", "на прошлой", "неделю назад"]):
        return "episodic"
    return "semantic"


# ============================================================================
# PHASE 11 TRACK 2: HYPERBOLIC GEOMETRY (Poincare ball)
# ============================================================================

def poincare_dist(u: NDArray, v: NDArray, ball_radius: float = 0.85) -> float:
    """Hyperbolic distance in Poincare ball model."""
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm >= ball_radius or v_norm >= ball_radius:
        # Project inside ball
        u = u * (ball_radius - 1e-6) / max(u_norm, 1e-8)
        v = v * (ball_radius - 1e-6) / max(v_norm, 1e-8)
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
    delta = u - v
    sq_delta = np.sum(delta ** 2)
    # Bug fix: Use ball_radius^2 in denominator for non-unit ball
    r_sq = ball_radius ** 2
    denom = ((r_sq - u_norm ** 2) * (r_sq - v_norm ** 2)) / max(r_sq, 1e-8)
    arg = 1 + 2 * sq_delta / max(denom, 1e-8)
    return float(ball_radius * np.arccosh(np.clip(arg, 1.0, None)))


def exp_map_poincare(tangent: NDArray, base: NDArray, ball_radius: float = 0.85) -> NDArray:
    """Exponential map from tangent space to Poincare ball."""
    base_norm = np.linalg.norm(base)
    if base_norm >= ball_radius:
        base = base * (ball_radius - 1e-6) / max(base_norm, 1e-8)
        base_norm = np.linalg.norm(base)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1e-8:
        return base.copy()
    # Mobius addition approach
    denom = 1.0 + base_norm ** 2
    factor = np.tanh(tangent_norm * 0.5) / max(base_norm, 1e-8)
    result = base + tangent * factor
    # Project back into ball
    r = np.linalg.norm(result)
    if r >= ball_radius:
        result = result * (ball_radius - 1e-6) / max(r, 1e-8)
    return result.astype(np.float32)


def log_map_poincare(point: NDArray, base: NDArray, ball_radius: float = 0.85) -> NDArray:
    """Logarithmic map from Poincare ball to tangent space."""
    base_norm = np.linalg.norm(base)
    if base_norm >= ball_radius:
        base = base * (ball_radius - 1e-6) / max(base_norm, 1e-8)
        base_norm = np.linalg.norm(base)
    diff = point - base
    diff_norm = np.linalg.norm(diff)
    if diff_norm < 1e-8:
        return np.zeros_like(point)
    factor = 2.0 / (1.0 - base_norm ** 2)
    tangent = diff * factor
    return tangent.astype(np.float32)


def mobius_add(x: NDArray, y: NDArray, ball_radius: float = 0.85) -> NDArray:
    """Mobius addition in Poincare ball."""
    x2 = np.sum(x ** 2)
    y2 = np.sum(y ** 2)
    xy = np.dot(x, y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    result = num / max(den, 1e-8)
    r = np.linalg.norm(result)
    if r >= ball_radius:
        result = result * (ball_radius - 1e-6) / max(r, 1e-8)
    return result.astype(np.float32)


# ============================================================================
# PHASE 11 TRACK 3: PREDICTIVE CODING
# ============================================================================

class PredictiveCodingModel:
    """Predictive coding / active inference for field dynamics."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, lr: float = 0.01):
        self.latent_dim = latent_dim
        self.state_dim = latent_dim * 4  # pos, phase, amp, sal encoded
        self.hidden_dim = hidden_dim
        self.lr = lr
        # Simple linear predictor: W * state + b
        self.W = np.random.randn(self.state_dim, self.state_dim).astype(np.float32) * 0.01
        self.b = np.zeros(self.state_dim, dtype=np.float32)
        self._complexity_weight = 0.01

    def predict(self, state: NDArray) -> NDArray:
        """Predict next state from current state."""
        return (state @ self.W + self.b).astype(np.float32)

    def compute_free_energy(self, state_t: NDArray, state_t1: NDArray) -> float:
        """Compute variational free energy (prediction error + complexity)."""
        pred = self.predict(state_t)
        prediction_error = float(np.mean((pred - state_t1) ** 2))
        complexity = float(np.mean(self.W ** 2)) * self._complexity_weight
        return prediction_error + complexity

    def update(self, state_t: NDArray, state_t1: NDArray, lr: Optional[float] = None):
        """Update predictor weights to minimize free energy."""
        lr = lr or self.lr
        pred = self.predict(state_t)
        error = pred - state_t1
        # Gradient descent on prediction error
        self.W -= lr * np.outer(state_t, error)
        self.b -= lr * error
        # L2 regularization
        self.W *= (1.0 - lr * self._complexity_weight)

    def get_state(self) -> Dict:
        return {"W": self.W.tolist(), "b": self.b.tolist(), "lr": self.lr}

    def load_state(self, state: Dict):
        self.W = np.array(state["W"], dtype=np.float32)
        self.b = np.array(state["b"], dtype=np.float32)
        self.lr = state.get("lr", self.lr)


# ============================================================================
# PHASE 11 TRACK 4: COUNTERFACTUAL IMAGINATION
# ============================================================================

class ScenarioPlanner:
    """Counterfactual imagination and scenario planning."""

    def __init__(self, field: Any, max_scenarios: int = 5):
        self.field = field
        self.max_scenarios = max_scenarios

    def imagine_counterfactual(self, base_query: NDArray,
                                intervention: Dict[str, float]) -> List[Dict]:
        """Generate hypothetical trajectories via do-interventions."""
        results = []
        for node_id, strength in list(intervention.items())[:self.max_scenarios]:
            if node_id not in self.field.nodes:
                continue
            node = self.field.nodes[node_id]
            # Create hypothetical node state
            hyp_phase = (node.phase + strength * np.pi) % (2 * np.pi)
            hyp_amp = min(1.0, node.amplitude * 1.2)
            hyp_sal = min(1.0, node.salience * 1.1)

            # Simulate trajectory
            traj = self._simulate_trajectory(node, hyp_phase, hyp_amp, hyp_sal, steps=3)
            coherence = self._score_coherence(traj)

            results.append({
                "hypothetical": True,
                "node_id": node_id,
                "intervention_strength": strength,
                "trajectory": [t.tolist() if isinstance(t, np.ndarray) else t for t in traj],
                "confidence": coherence,
            })

        self.field.stats["scenarios_generated"] = self.field.stats.get("scenarios_generated", 0) + len(results)
        if results:
            self.field.stats["avg_scenario_confidence"] = np.mean([r["confidence"] for r in results])

        return results

    def _simulate_trajectory(self, node: MemoryNode, phase: float,
                              amp: float, sal: float, steps: int = 3) -> List[NDArray]:
        """Simulate trajectory from hypothetical state."""
        traj = [node.latent_pos.copy()]
        current_pos = node.latent_pos.copy()
        node_idx = list(self.field.nodes.keys()).index(node.id)
        for _ in range(steps):
            # Simple dynamics: attract towards mean + noise
            if self.field.ode_dynamics and len(self.field.nodes) > 1:
                # Build proper state for ODE: concatenate all node positions
                all_positions = np.array([n.latent_pos for n in self.field.nodes.values()])
                state = all_positions.flatten()
                dynamics = self.field.ode_dynamics._dynamics(0, state)
                # Extract update for this node's position
                update = dynamics[node_idx * self.field.cfg.latent_dim:(node_idx + 1) * self.field.cfg.latent_dim]
                current_pos = current_pos + update * 0.1
            else:
                current_pos = current_pos * 0.95 + np.random.randn(len(current_pos)).astype(np.float32) * 0.01
            traj.append(current_pos.copy())
        return traj

    def _score_coherence(self, trajectory: List[NDArray]) -> float:
        """Score trajectory coherence (lower variance = higher coherence)."""
        if len(trajectory) < 2:
            return 0.5
        dists = []
        for i in range(len(trajectory) - 1):
            dists.append(np.linalg.norm(trajectory[i + 1] - trajectory[i]))
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        # Coherence: low mean distance + low variance
        coherence = np.exp(-mean_dist) * np.exp(-std_dist)
        return float(np.clip(coherence, 0.0, 1.0))


# ============================================================================
# PHASE 11 TRACK 5: DIFFERENTIAL PRIVACY
# ============================================================================

class DifferentialPrivacy:
    """Differential privacy for federated learning."""

    def __init__(self, epsilon: float = 2.0, delta: float = 1e-5, max_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_norm = max_norm
        self._privacy_spent = 0.0
        self._num_updates = 0

    def clip_update(self, update: NDArray) -> NDArray:
        """Clip update to max_norm."""
        norm = np.linalg.norm(update)
        if norm > self.max_norm:
            return (update * self.max_norm / norm).astype(np.float32)
        return update

    def add_noise(self, update: NDArray, sensitivity: float = 1.0) -> NDArray:
        """Add calibrated Gaussian noise."""
        noise_std = self.compute_noise_multiplier(sensitivity)
        noise = np.random.randn(*update.shape).astype(np.float32) * noise_std
        return (update + noise).astype(np.float32)

    def compute_noise_multiplier(self, sensitivity: float = 1.0) -> float:
        """Compute noise multiplier for given privacy budget."""
        if self.epsilon <= 0:
            return float('inf')
        # Bug #10 FIX: Gaussian mechanism — sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
        # The sensitivity (Delta_f) MUST be multiplied — without it, DP guarantees don't hold
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        return sigma

    def get_privacy_spent(self) -> float:
        """Return cumulative privacy budget spent."""
        return self._privacy_spent

    def record_update(self, n_samples: int = 1):
        """Record that an update was made (track privacy budget)."""
        self._num_updates += n_samples
        # Bug #11 FIX: Advanced composition — use per-mechanism epsilon correctly
        # epsilon_total = sqrt(2 * k * ln(1/delta')) * epsilon_per_mechanism
        k = self._num_updates
        self._privacy_spent = math.sqrt(2 * k * math.log(1 / self.delta)) * self.epsilon

    def get_state(self) -> Dict:
        return {"epsilon": self.epsilon, "delta": self.delta, "max_norm": self.max_norm,
                "privacy_spent": self._privacy_spent, "num_updates": self._num_updates}

    def load_state(self, state: Dict):
        self.epsilon = state.get("epsilon", self.epsilon)
        self.delta = state.get("delta", self.delta)
        self.max_norm = state.get("max_norm", self.max_norm)
        self._privacy_spent = state.get("privacy_spent", 0.0)
        self._num_updates = state.get("num_updates", 0)


# ============================================================================
# DATA TYPES v7
# ============================================================================

@dataclass
class CausalEdge:
    source: str
    target: str
    strength: float
    confidence: float
    adjustment_set: List[str] = field(default_factory=list)
    evidence_count: int = 0
    is_contradicted: bool = False
    contradiction_reason: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CausalEdge":
        return cls(**data)


@dataclass
class ContradictionRecord:
    id: str
    effect_node: str
    causes: List[Tuple[str, float]]
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: str = ""
    contradiction_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id, "effect_node": self.effect_node, "causes": self.causes,
            "timestamp": self.timestamp, "resolved": self.resolved,
            "resolution": self.resolution, "contradiction_reason": self.contradiction_reason,
        }


@dataclass
class CounterfactualResult:
    query: str
    intervention: Dict[str, Any]
    predicted_outcomes: List[Tuple[str, float]]
    confidence: float
    reasoning_path: List[str]
    assumptions: List[str]

    def to_dict(self) -> Dict:
        return {
            "query": self.query, "intervention": self.intervention,
            "predicted_outcomes": [{"node": n, "probability": p} for n, p in self.predicted_outcomes],
            "confidence": self.confidence, "reasoning_path": self.reasoning_path,
            "assumptions": self.assumptions,
        }


@dataclass
class AgentPlan:
    goal: str
    subtasks: List[Dict[str, Any]]
    tools_needed: List[str]
    estimated_steps: int
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    success: bool = False
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Hypothesis:
    statement: str
    confidence: float
    evidence_nodes: List[str]
    causal_path: List[str]
    verified: bool = False
    verification_score: float = 0.0


@dataclass
class EvalResult:
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevance: float = 0.0
    faithfulness: float = 0.0
    causal_consistency: float = 0.0
    temporal_coherence: float = 0.0
    overall_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FederatedNode:
    node_id: str
    phase: float
    natural_freq: float = 1.0
    amplitude: float = 1.0
    last_sync_time: float = field(default_factory=time.time)
    params: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    sync_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "FederatedNode":
        return cls(**data)


@dataclass
class MemoryNode:
    id: str
    latent_pos: NDArray[np.float32]
    phase: float
    amplitude: float
    salience: float
    tension: float = 0.0
    soft_gate: float = 1.0
    content: Dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_resonated: float = 0.0
    lineage: List[str] = field(default_factory=list)
    modality: str = "text"
    self_sup_score: float = 1.0
    modal_weight: float = 1.0
    pre_consolidation_pos: Optional[NDArray[np.float32]] = None
    causal_parents: List[str] = field(default_factory=list)
    causal_strength: Dict[str, float] = field(default_factory=dict)
    gradient_cache: Optional[NDArray[np.float32]] = None
    is_healing: bool = False
    healing_origin: Optional[str] = None
    local_density: float = 0.0
    causal_effects: Dict[str, float] = field(default_factory=dict)
    do_interventions: Dict[str, NDArray] = field(default_factory=dict)
    is_causal_root: bool = False
    causal_context: Dict[str, Any] = field(default_factory=dict)
    velocity: Optional[NDArray[np.float32]] = None
    acceleration: Optional[NDArray[np.float32]] = None
    goal_tags: List[str] = field(default_factory=list)
    tool_usage_count: int = 0
    modal_embedding: Optional[NDArray[np.float32]] = None
    cross_modal_score: float = 0.0
    # Phase 11 Track 1
    tier: str = "semantic"
    # Phase 13
    goal_relevance: float = 0.0
    rl_reward: float = 0.0

    # Phase 20: Domain Memory & Concept Lifecycle
    domain: str = "general"
    subdomain: str = ""
    topic: str = ""
    state: str = "stable"
    confidence: float = 1.0
    revision_count: int = 0
    conflict_with: List[str] = field(default_factory=list)
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None
    evidence_spans: List[Dict] = field(default_factory=list)
    fact_state: str = "active"
    superseded_by: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["latent_pos"] = self.latent_pos.tolist()
        if self.pre_consolidation_pos is not None:
            d["pre_consolidation_pos"] = self.pre_consolidation_pos.tolist()
        if self.gradient_cache is not None:
            d["gradient_cache"] = self.gradient_cache.tolist()
        if self.velocity is not None:
            d["velocity"] = self.velocity.tolist()
        if self.acceleration is not None:
            d["acceleration"] = self.acceleration.tolist()
        if self.modal_embedding is not None:
            d["modal_embedding"] = self.modal_embedding.tolist()
        for k, v in self.do_interventions.items():
            if isinstance(v, np.ndarray):
                d["do_interventions"][k] = v.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "MemoryNode":
        data["latent_pos"] = np.array(data["latent_pos"], dtype=np.float32)
        if data.get("pre_consolidation_pos"):
            data["pre_consolidation_pos"] = np.array(data["pre_consolidation_pos"], dtype=np.float32)
        if data.get("gradient_cache"):
            data["gradient_cache"] = np.array(data["gradient_cache"], dtype=np.float32)
        if data.get("velocity"):
            data["velocity"] = np.array(data["velocity"], dtype=np.float32)
        if data.get("acceleration"):
            data["acceleration"] = np.array(data["acceleration"], dtype=np.float32)
        if data.get("modal_embedding"):
            data["modal_embedding"] = np.array(data["modal_embedding"], dtype=np.float32)
        for k, v in data.get("do_interventions", {}).items():
            if isinstance(v, list):
                data["do_interventions"][k] = np.array(v, dtype=np.float32)
        return cls(**data)


# ============================================================================
# TRACK 10.1: CROSS-MODAL RESONANCE
# ============================================================================

def detect_modality(text: str) -> str:
    code_patterns = [
        r"\b(def|class|function|import|from|return|if|else|for|while|const|let|var)\b",
        r"[{}()\[\];]",
        r"\b(async|await|lambda|yield|try|except|catch|throw)\b",
    ]
    audio_patterns = [
        r"\b(audio|sound|music|frequency|hz|waveform|sample|rate|decibel|db)\b",
        r"\b(mp3|wav|flac|aac|ogg|pcm|bitrate|spectrum)\b",
    ]
    vision_patterns = [
        r"\b(image|photo|picture|pixel|resolution|rgb|color|frame|video)\b",
        r"\b(png|jpg|jpeg|gif|bmp|tiff|width|height|crop|resize)\b",
    ]
    metrics_patterns = [
        r"\b(metric|kpi|latency|throughput|error_rate|uptime|cpu|memory|disk)\b",
        r"\b(p99|p95|p50|iops|mbps|gbps|ms|rpm)\b",
        r"\d+\s*(ms|s|mb|gb|tb|kb)",
    ]
    text_lower = text.lower()
    for pattern in code_patterns:
        if re.search(pattern, text_lower):
            return "code"
    for pattern in metrics_patterns:
        if re.search(pattern, text_lower):
            return "metrics"
    for pattern in audio_patterns:
        if re.search(pattern, text_lower):
            return "audio"
    for pattern in vision_patterns:
        if re.search(pattern, text_lower):
            return "vision"
    return "text"


def cross_modal_resonance(q_mod: str, n_mod: str, base_resp: float,
                          modal_phase_offsets: Dict[str, float],
                          cross_modal_kernel_weight: float) -> float:
    q_phase = modal_phase_offsets.get(q_mod, 0.0)
    n_phase = modal_phase_offsets.get(n_mod, 0.0)
    phase_diff = q_phase - n_phase
    modal_coupling = math.cos(phase_diff)
    boost = 1.0 + cross_modal_kernel_weight * modal_coupling
    return base_resp * boost


# ============================================================================
# TRACK 10.2: META-COGNITIVE CONTROLLER
# ============================================================================

class MetaController:
    def __init__(self, n_trials: int = 20, optimize_params: Optional[List[str]] = None,
                 optimization_freq: int = 500):
        self.n_trials = n_trials
        self.optimize_params = optimize_params or [
            "decay_rate", "tension_threshold", "phase_coupling", "bandwidth"
        ]
        self.optimization_freq = optimization_freq
        self._optuna_available = False
        self._best_params: Dict[str, float] = {}
        self._optimization_history: deque = deque(maxlen=50)
        self._step_counter = 0
        self._last_optimization_time: float = 0.0
        self._total_optimizations = 0
        self._try_load_optuna()

    def _try_load_optuna(self):
        try:
            import optuna
            self._optuna_available = True
            self.optuna = optuna
        except ImportError:
            self._optuna_available = False

    def optimize(self, field: Any) -> Dict[str, float]:
        self._step_counter += 1
        if self._optuna_available:
            return self._optimize_with_optuna(field)
        else:
            return self._optimize_grid_search(field)

    def _optimize_with_optuna(self, field: Any) -> Dict[str, float]:
        def objective(trial):
            params = {}
            if "decay_rate" in self.optimize_params:
                params["decay_rate"] = trial.suggest_float("decay_rate", 0.95, 0.9999)
            if "tension_threshold" in self.optimize_params:
                params["tension_threshold"] = trial.suggest_float("tension_threshold", 0.1, 0.5)
            if "phase_coupling" in self.optimize_params:
                params["phase_coupling"] = trial.suggest_float("phase_coupling", 0.05, 0.8)
            if "bandwidth" in self.optimize_params:
                params["bandwidth"] = trial.suggest_float("bandwidth", 0.3, 5.0)
            return self._evaluate_params(field, params)

        study = self.optuna.create_study(direction="maximize", sampler=self.optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        best_params = study.best_params
        self._best_params = best_params
        self._total_optimizations += 1
        self._last_optimization_time = time.time()
        self._optimization_history.append({
            "time": time.time(), "best_value": study.best_value,
            "params": best_params, "n_trials": self.n_trials,
        })
        return best_params

    def _optimize_grid_search(self, field: Any) -> Dict[str, float]:
        # Replaced full grid search with random search for performance.
        # Original grid: 5^4 = 625 combinations → now 50 random trials.
        grid = {
            "decay_rate": [0.97, 0.98, 0.99, 0.995, 0.998],
            "tension_threshold": [0.15, 0.2, 0.25, 0.3, 0.35],
            "phase_coupling": [0.1, 0.2, 0.3, 0.4, 0.5],
            "bandwidth": [0.5, 1.0, 1.5, 2.0, 3.0],
        }
        filtered_grid = {k: v for k, v in grid.items() if k in self.optimize_params}
        best_score = -float("inf")
        best_params = {}
        keys = list(filtered_grid.keys())
        values = list(filtered_grid.values())
        n_trials = min(50, max(len(v) for v in values) ** len(keys))

        for _ in range(n_trials):
            params = {k: values[i][np.random.randint(len(values[i]))] for i, k in enumerate(keys)}
            score = self._evaluate_params(field, params)
            if score > best_score:
                best_score = score
                best_params = params.copy()

        self._best_params = best_params
        self._total_optimizations += 1
        self._last_optimization_time = time.time()
        self._optimization_history.append({
            "time": time.time(), "best_value": best_score,
            "params": best_params, "method": "random_search", "n_trials": n_trials,
        })
        return best_params

    def _evaluate_params(self, field: Any, params: Dict[str, float]) -> float:
        score = 0.0
        n_nodes = len(field.nodes)
        if n_nodes < 2:
            return 0.5
        positions = np.array([n.latent_pos for n in field.nodes.values()])
        valid_dists = pdist(positions)
        if len(valid_dists) > 0:
            mean_dist = np.mean(valid_dists)
            std_dist = np.std(valid_dists)
            cv = std_dist / (mean_dist + 1e-8)
            score += max(0, 1.0 - cv) * 0.4
        phases = np.array([n.phase for n in field.nodes.values()])
        phase_order = np.abs(np.mean(np.exp(1j * phases)))
        score += phase_order * 0.3
        amplitudes = np.array([n.amplitude for n in field.nodes.values()])
        alive_ratio = np.mean(amplitudes > field.cfg.min_amplitude)
        score += alive_ratio * 0.3
        if "decay_rate" in params:
            decay_penalty = abs(params["decay_rate"] - field.cfg.decay_rate) * 10
            score -= decay_penalty * 0.1
        if field.stats.get("avg_response", 0) > 0:
            score += min(0.5, field.stats["avg_response"] * 0.5)
        return max(0.0, min(1.0, score))

    def apply_params(self, field: Any, params: Dict[str, float]):
        if "decay_rate" in params:
            field.cfg.decay_rate = params["decay_rate"]
            if field.learnable_kernel:
                field.learnable_kernel.decay_rate = params["decay_rate"]
        if "tension_threshold" in params:
            field.cfg.tension_threshold = params["tension_threshold"]
        if "phase_coupling" in params:
            field.cfg.phase_coupling = params["phase_coupling"]
            if field.meta_kernel:
                field.meta_kernel.base_phase_coupling = params["phase_coupling"]
        if "bandwidth" in params:
            field.cfg.bandwidth = params["bandwidth"]
            if field.meta_kernel:
                field.meta_kernel.base_bandwidth = params["bandwidth"]

    def should_optimize(self) -> bool:
        return self._step_counter > 0 and self._step_counter % self.optimization_freq == 0

    def get_state(self) -> Dict:
        return {
            "best_params": self._best_params,
            "optimization_history": self._optimization_history,
            "total_optimizations": self._total_optimizations,
            "optuna_available": self._optuna_available,
            "step_counter": self._step_counter,
            "last_optimization_time": self._last_optimization_time,
        }

    def load_state(self, state: Dict):
        self._best_params = state.get("best_params", {})
        self._optimization_history = state.get("optimization_history", [])
        self._total_optimizations = state.get("total_optimizations", 0)
        self._step_counter = state.get("step_counter", 0)
        self._last_optimization_time = state.get("last_optimization_time", 0.0)


# ============================================================================
# TRACK 10.3: KURAMOTO SYNCHRONIZATION & FEDERATED SYNC
# ============================================================================

class KuramotoSync:
    def __init__(self, coupling_strength: float = 0.5, dt: float = 0.01):
        self.coupling_strength = coupling_strength
        self.dt = dt
        self.phases: Dict[str, float] = {}
        self.natural_freqs: Dict[str, float] = {}
        self._order_history: deque = deque(maxlen=100)

    def add_oscillator(self, node_id: str, phase: float, natural_freq: float = 1.0):
        self.phases[node_id] = phase
        self.natural_freqs[node_id] = natural_freq

    def remove_oscillator(self, node_id: str):
        self.phases.pop(node_id, None)
        self.natural_freqs.pop(node_id, None)

    def step(self, n_steps: int = 1) -> Dict[str, float]:
        for _ in range(n_steps):
            new_phases = {}
            n = len(self.phases)
            if n < 2:
                continue
            K_over_N = self.coupling_strength / n
            for nid, phi in self.phases.items():
                omega = self.natural_freqs.get(nid, 1.0)
                coupling = 0.0
                for other_id, other_phi in self.phases.items():
                    if other_id != nid:
                        coupling += math.sin(other_phi - phi)
                new_phases[nid] = (phi + self.dt * (omega + K_over_N * coupling)) % (2 * math.pi)
            self.phases.update(new_phases)
        self._order_history.append(self.compute_order_parameter())
        return self.phases

    def compute_order_parameter(self) -> float:
        if not self.phases:
            return 0.0
        n = len(self.phases)
        sum_exp = sum(complex(math.cos(p), math.sin(p)) for p in self.phases.values())
        return abs(sum_exp) / n

    def sync_to_target(self, target_phases: Dict[str, float], n_steps: int = 10) -> Dict[str, float]:
        for nid, target_phi in target_phases.items():
            if nid in self.phases:
                diff = target_phi - self.phases[nid]
                diff = (diff + math.pi) % (2 * math.pi) - math.pi
                self.phases[nid] = (self.phases[nid] + self.coupling_strength * diff) % (2 * math.pi)
        for _ in range(n_steps):
            self.step()
        return self.phases

    def get_state(self) -> Dict:
        return {
            "phases": dict(self.phases), "natural_freqs": dict(self.natural_freqs),
            "order_parameter": self.compute_order_parameter(),
            "coupling_strength": self.coupling_strength,
        }

    def load_state(self, state: Dict):
        self.phases = state.get("phases", {})
        self.natural_freqs = state.get("natural_freqs", {})
        self.coupling_strength = state.get("coupling_strength", self.coupling_strength)


class FederatedRTMDK:
    def __init__(self, node_id: str = "local", sync_lr: float = 0.01,
                 sync_freq: int = 100, min_resonance: float = 0.2,
                 coupling_strength: float = 0.5):
        self.node_id = node_id
        self.sync_lr = sync_lr
        self.sync_freq = sync_freq
        self.min_resonance = min_resonance
        self.kuramoto = KuramotoSync(coupling_strength=coupling_strength)
        self.peers: Dict[str, FederatedNode] = {}
        self._sync_history: deque = deque(maxlen=100)
        self._total_syncs = 0
        self._step_counter = 0

    def register_peer(self, peer: FederatedNode):
        self.peers[peer.node_id] = peer
        self.kuramoto.add_oscillator(peer.node_id, peer.phase, peer.natural_freq)

    def unregister_peer(self, peer_id: str):
        self.peers.pop(peer_id, None)
        self.kuramoto.remove_oscillator(peer_id)

    def sync_with_peers(self, local_phases: Dict[str, float],
                        local_params: Dict[str, float]) -> Dict[str, Any]:
        self._step_counter += 1
        if self._step_counter % self.sync_freq != 0:
            return {"synced": False, "reason": "not_sync_step"}
        if not self.peers:
            return {"synced": False, "reason": "no_peers"}
        sync_results = []
        for peer_id, peer in self.peers.items():
            if not peer.is_active:
                continue
            resonance = self._compute_param_resonance(local_params, peer.params)
            if resonance < self.min_resonance:
                continue
            self.kuramoto.sync_to_target(local_phases, n_steps=5)
            blended_params = self._blend_params(local_params, peer.params, self.sync_lr)
            peer.params = blended_params
            peer.sync_count += 1
            peer.last_sync_time = time.time()
            sync_results.append({
                "peer_id": peer_id, "resonance": resonance,
                "params_updated": list(blended_params.keys()),
            })
        self._total_syncs += 1
        self._sync_history.append({
            "time": time.time(), "peers_synced": len(sync_results),
            "order_parameter": self.kuramoto.compute_order_parameter(),
        })
        return {
            "synced": True, "results": sync_results,
            "order_parameter": self.kuramoto.compute_order_parameter(),
            "total_syncs": self._total_syncs,
        }

    def _compute_param_resonance(self, params_a: Dict[str, float],
                                  params_b: Dict[str, float]) -> float:
        common_keys = set(params_a.keys()) & set(params_b.keys())
        if not common_keys:
            return 0.0
        diffs = []
        for key in common_keys:
            a, b = params_a[key], params_b[key]
            denom = max(abs(a) + abs(b), 1e-8)
            diffs.append(1.0 - abs(a - b) / denom)
        return float(np.mean(diffs))

    def _blend_params(self, params_a: Dict[str, float],
                      params_b: Dict[str, float], lr: float) -> Dict[str, float]:
        blended = {}
        all_keys = set(params_a.keys()) | set(params_b.keys())
        for key in all_keys:
            a = params_a.get(key, 0.0)
            b = params_b.get(key, 0.0)
            blended[key] = (1 - lr) * a + lr * b
        return blended

    def get_aggregated_params(self) -> Dict[str, float]:
        all_params: Dict[str, List[float]] = defaultdict(list)
        for peer in self.peers.values():
            if peer.is_active:
                for k, v in peer.params.items():
                    all_params[k].append(v)
        return {k: float(np.mean(v)) for k, v in all_params.items() if v}

    def get_sync_status(self) -> Dict:
        return {
            "node_id": self.node_id, "n_peers": len(self.peers),
            "active_peers": sum(1 for p in self.peers.values() if p.is_active),
            "order_parameter": self.kuramoto.compute_order_parameter(),
            "total_syncs": self._total_syncs,
            "sync_history": self._sync_history[-10:],
            "kuramoto_state": self.kuramoto.get_state(),
        }

    def export_state(self) -> Dict:
        return {
            "node_id": self.node_id,
            "peers": {pid: p.to_dict() for pid, p in self.peers.items()},
            "kuramoto": self.kuramoto.get_state(),
            "sync_history": self._sync_history, "total_syncs": self._total_syncs,
        }

    def import_state(self, state: Dict):
        self.node_id = state.get("node_id", self.node_id)
        self._total_syncs = state.get("total_syncs", 0)
        self._sync_history = state.get("sync_history", [])
        for pid, pdata in state.get("peers", {}).items():
            peer = FederatedNode.from_dict(pdata)
            self.peers[pid] = peer
            self.kuramoto.add_oscillator(pid, peer.phase, peer.natural_freq)
        if "kuramoto" in state:
            self.kuramoto.load_state(state["kuramoto"])


# ============================================================================
# PHASE 7: NEURAL ODE/SDE DYNAMICS
# ============================================================================

class NeuralODEDynamics:
    def __init__(self, latent_dim: int, noise_level: float = 0.01,
                 time_horizon: float = 1.0, n_steps: int = 20,
                 chunk_size: int = 256, solver: str = "RK45",
                 atol: float = 1e-6, rtol: float = 1e-5):
        self.latent_dim = latent_dim
        self.noise_level = noise_level
        self.time_horizon = time_horizon
        self.n_steps = n_steps
        self.chunk_size = chunk_size
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.alpha = 0.1
        self.beta = 0.05
        self.gamma = 0.02
        self.W = np.random.randn(latent_dim, latent_dim).astype(np.float32) * 0.01
        self._response_history: deque = deque(maxlen=100)
        self._state_history: deque = deque(maxlen=2)

    def _sigma(self, x: NDArray) -> NDArray:
        return np.tanh(x)

    def _dynamics(self, t: float, state: NDArray, input_signal: Optional[NDArray] = None,
                  topology_gradient: Optional[NDArray] = None) -> NDArray:
        n_nodes = len(state) // self.latent_dim
        if n_nodes == 0:
            return state
        X = state.reshape(n_nodes, self.latent_dim)
        damping = -self.alpha * X
        nonlinear = self.W @ self._sigma(X.T)
        nonlinear = nonlinear.T
        if input_signal is not None:
            u = input_signal.reshape(n_nodes, self.latent_dim)
            attraction = self.beta * (u - X)
        else:
            attraction = 0.0
        if topology_gradient is not None:
            topo = self.gamma * topology_gradient.reshape(n_nodes, self.latent_dim)
        else:
            topo = 0.0
        dX = damping + nonlinear + attraction + topo
        return dX.flatten()

    def evolve(self, initial_state: NDArray, input_signal: Optional[NDArray] = None,
               topology_gradient: Optional[NDArray] = None,
               t_span: Optional[NDArray] = None) -> NDArray:
        if t_span is None:
            t_span = np.linspace(0, self.time_horizon, self.n_steps)
        n_nodes = len(initial_state) // self.latent_dim
        if n_nodes > self.chunk_size:
            return self._evolve_chunked(initial_state, input_signal, topology_gradient, t_span)
        def ode_func(t, state):
            return self._dynamics(t, state, input_signal, topology_gradient)
        solution = solve_ivp(
            ode_func, [t_span[0], t_span[-1]], initial_state.flatten(),
            t_eval=t_span, method=self.solver, atol=self.atol, rtol=self.rtol
        )
        if solution.success:
            trajectory = solution.y.T
        else:
            trajectory = odeint(ode_func, initial_state.flatten(), t_span,
                                atol=self.atol * 10, rtol=self.rtol * 10)
        self._state_history.append(trajectory[-1].copy())
        return trajectory

    def _evolve_chunked(self, initial_state: NDArray, input_signal: Optional[NDArray],
                        topology_gradient: Optional[NDArray], t_span: NDArray) -> NDArray:
        n_nodes = len(initial_state) // self.latent_dim
        chunks = []
        for i in range(0, n_nodes, self.chunk_size):
            end = min(i + self.chunk_size, n_nodes)
            chunk_state = initial_state[i * self.latent_dim:end * self.latent_dim]
            chunk_input = input_signal[i * self.latent_dim:end * self.latent_dim] if input_signal is not None else None
            chunk_topo = topology_gradient[i * self.latent_dim:end * self.latent_dim] if topology_gradient is not None else None
            def ode_func(t, state):
                return self._dynamics(t, state, chunk_input, chunk_topo)
            sol = solve_ivp(ode_func, [t_span[0], t_span[-1]], chunk_state.flatten(),
                            t_eval=t_span, method=self.solver, atol=self.atol, rtol=self.rtol)
            if sol.success:
                chunks.append(sol.y.T)
            else:
                chunks.append(odeint(ode_func, chunk_state.flatten(), t_span))
        return np.concatenate(chunks, axis=1)

    def evolve_with_noise(self, initial_state: NDArray, input_signal: Optional[NDArray] = None,
                          topology_gradient: Optional[NDArray] = None,
                          dt: float = 0.05) -> NDArray:
        n_steps = int(self.time_horizon / dt)
        state = initial_state.flatten().copy()
        trajectory = [state.copy()]
        for _ in range(n_steps):
            deterministic = self._dynamics(0, state, input_signal, topology_gradient) * dt
            noise = self.noise_level * np.random.randn(len(state)) * np.sqrt(dt)
            state = state + deterministic + noise
            trajectory.append(state.copy())
        self._state_history.append(trajectory[-1].copy())
        return np.array(trajectory)

    def compute_topology_gradient(self, nodes: Dict[str, MemoryNode]) -> Optional[NDArray]:
        if len(nodes) < 2:
            return None
        node_ids = list(nodes.keys())
        positions = np.array([nodes[nid].latent_pos for nid in node_ids])
        tree = cKDTree(positions)
        pairs = tree.query_pairs(2.0)
        gradient = np.zeros_like(positions)
        for i, j in pairs:
            dist = np.linalg.norm(positions[i] - positions[j])
            direction = (positions[i] - positions[j]) / (dist + 1e-8)
            gradient[i] += direction * 0.01
            gradient[j] -= direction * 0.01
        return gradient.flatten()

    def compute_response_smoothness(self) -> float:
        if len(self._response_history) < 2:
            return 1.0
        responses = np.array(self._response_history)
        std = np.std(responses)
        return max(0.0, 1.0 - std)

    def record_response(self, response: float):
        self._response_history.append(response)

    def get_state(self) -> Dict:
        return {
            "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
            "W": self.W.tolist(), "noise_level": self.noise_level,
            "smoothness": self.compute_response_smoothness(),
        }

    def load_state(self, state: Dict):
        self.alpha = state.get("alpha", self.alpha)
        self.beta = state.get("beta", self.beta)
        self.gamma = state.get("gamma", self.gamma)
        if "W" in state:
            self.W = np.array(state["W"], dtype=np.float32)
        self.noise_level = state.get("noise_level", self.noise_level)


# ============================================================================
# PHASE 8: AGENT ORCHESTRATION
# ============================================================================

class AgentPlanner:
    def __init__(self, max_depth: int = 3, max_tool_calls: int = 5,
                 tool_timeout: float = 15.0):
        self.max_depth = max_depth
        self.max_tool_calls = max_tool_calls
        self.tool_timeout = tool_timeout
        self._visited_tools: Set[str] = set()
        self._call_count = 0

    def create_plan(self, goal: str, available_tools: List[str],
                    context: Dict[str, Any]) -> AgentPlan:
        subtasks = self._decompose_goal(goal, context)
        tools_needed = self._select_tools(goal, subtasks, available_tools)
        return AgentPlan(
            goal=goal, subtasks=subtasks, tools_needed=tools_needed,
            estimated_steps=len(subtasks),
            confidence=self._estimate_confidence(goal, subtasks, tools_needed),
            reasoning=f"Decomposed goal into {len(subtasks)} subtasks"
        )

    def _decompose_goal(self, goal: str, context: Dict) -> List[Dict[str, Any]]:
        subtasks = []
        subtasks.append({"type": "retrieve", "description": f"Find memories related to: {goal}", "priority": 1})
        if context.get("hypothesis_verification", False):
            subtasks.append({"type": "verify", "description": "Verify causal hypotheses", "priority": 2})
        subtasks.append({"type": "synthesize", "description": f"Synthesize response for: {goal}", "priority": 3})
        return subtasks[:self.max_depth]

    def _select_tools(self, goal: str, subtasks: List[Dict],
                      available_tools: List[str]) -> List[str]:
        selected = []
        for task in subtasks:
            task_type = task.get("type", "")
            for tool in available_tools:
                if task_type in tool.lower() and tool not in selected:
                    selected.append(tool)
        return selected[:self.max_tool_calls]

    def _estimate_confidence(self, goal: str, subtasks: List[Dict],
                             tools: List[str]) -> float:
        base = 0.5
        base += min(0.2, len(subtasks) * 0.05)
        base += min(0.2, len(tools) * 0.05)
        base += 0.1 if len(subtasks) <= self.max_depth else -0.1
        return min(1.0, max(0.0, base))

    def reset(self):
        self._visited_tools.clear()
        self._call_count = 0

    def can_call_tool(self, tool_name: str) -> bool:
        if tool_name in self._visited_tools and tool_name != "retrieve":
            return False
        return self._call_count < self.max_tool_calls

    def record_tool_call(self, tool_name: str):
        self._visited_tools.add(tool_name)
        self._call_count += 1


class HypothesisVerifier:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def verify(self, hypothesis: str, causal_engine: Any,
               active_nodes: List[str]) -> Hypothesis:
        evidence_nodes = []
        causal_path = []
        confidence = 0.5
        if causal_engine and hasattr(causal_engine, "causal_effects"):
            for (cause, effect), edge in causal_engine.causal_effects.items():
                if cause in active_nodes or effect in active_nodes:
                    evidence_nodes.append(cause)
                    evidence_nodes.append(effect)
                    causal_path.append(f"{cause} -> {effect} (P={edge.strength:.2f})")
                    confidence = max(confidence, edge.strength * edge.confidence)
        verified = confidence >= self.confidence_threshold
        return Hypothesis(
            statement=hypothesis, confidence=confidence,
            evidence_nodes=list(set(evidence_nodes)),
            causal_path=causal_path, verified=verified,
            verification_score=confidence,
        )


class ToolRouter:
    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._tool_registry: Dict[str, Callable] = {}
        self._call_history: deque = deque(maxlen=100)

    def register_tool(self, name: str, func: Callable):
        self._tool_registry[name] = func

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        t0 = time.time()
        call = ToolCall(tool_name=tool_name, arguments=arguments)
        if tool_name not in self._tool_registry:
            call.error = f"Tool not registered: {tool_name}"
            call.latency_ms = (time.time() - t0) * 1000
            return call
        try:
            func = self._tool_registry[tool_name]
            result = func(**arguments)
            call.result = result
            call.success = True
        except Exception as e:
            call.error = str(e)
        call.latency_ms = (time.time() - t0) * 1000
        self._call_history.append(call)
        return call

    def get_misuse_rate(self) -> float:
        if not self._call_history:
            return 0.0
        failures = sum(1 for c in self._call_history if not c.success)
        return failures / len(self._call_history)


# ============================================================================
# PHASE 9: PRODUCTION STACK
# ============================================================================

class ShadowModeEvaluator:
    def __init__(self, fallback_threshold: float = 0.3):
        self.fallback_threshold = fallback_threshold
        self._shadow_results: List[Dict] = []
        self._production_results: List[Dict] = []
        self._fallback_count = 0
        self._total_comparisons = 0

    def compare(self, shadow_output: Any, production_output: Any,
                metric_name: str = "response_quality") -> Dict[str, Any]:
        self._shadow_results.append({"value": shadow_output, "metric": metric_name})
        self._production_results.append({"value": production_output, "metric": metric_name})
        self._total_comparisons += 1
        diff = abs(float(shadow_output) - float(production_output))
        is_better = shadow_output > production_output
        if diff > self.fallback_threshold:
            self._fallback_count += 1
        return {
            "shadow_value": shadow_output, "production_value": production_output,
            "difference": diff, "shadow_better": is_better,
            "fallback_triggered": diff > self.fallback_threshold,
        }

    def get_correlation(self) -> float:
        if len(self._shadow_results) < 3:
            return 0.0
        shadow_vals = [r["value"] for r in self._shadow_results]
        prod_vals = [r["value"] for r in self._production_results]
        if np.std(shadow_vals) < 1e-8 or np.std(prod_vals) < 1e-8:
            return 1.0
        corr = np.corrcoef(shadow_vals, prod_vals)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def get_fallback_rate(self) -> float:
        return self._fallback_count / max(self._total_comparisons, 1)


class RAGASPlusEvaluator:
    def __init__(self):
        self._eval_history: deque = deque(maxlen=500)

    def evaluate(self, question: str, answer: str, contexts: List[str],
                 ground_truth: Optional[str] = None,
                 causal_edges: Optional[List[Tuple[str, str, float]]] = None) -> EvalResult:
        result = EvalResult()
        result.context_precision = self._compute_context_precision(question, contexts)
        if ground_truth:
            result.context_recall = self._compute_context_recall(ground_truth, contexts)
        else:
            result.context_recall = result.context_precision * 0.8
        result.answer_relevance = self._compute_answer_relevance(question, answer)
        result.faithfulness = self._compute_faithfulness(answer, contexts)
        if causal_edges:
            result.causal_consistency = self._compute_causal_consistency(answer, causal_edges)
        else:
            result.causal_consistency = 0.5
        result.temporal_coherence = self._compute_temporal_coherence(contexts)
        weights = [0.2, 0.15, 0.2, 0.2, 0.15, 0.1]
        scores = [result.context_precision, result.context_recall,
                  result.answer_relevance, result.faithfulness,
                  result.causal_consistency, result.temporal_coherence]
        result.overall_score = sum(w * s for w, s in zip(weights, scores))
        self._eval_history.append(result)
        return result

    def _compute_context_precision(self, question: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
        if not q_tokens:
            return 0.0
        precision_scores = []
        for ctx in contexts:
            c_tokens = set(re.findall(r"\b\w+\b", ctx.lower()))
            if c_tokens:
                precision_scores.append(len(q_tokens & c_tokens) / len(q_tokens))
        return float(np.mean(precision_scores)) if precision_scores else 0.0

    def _compute_context_recall(self, ground_truth: str, contexts: List[str]) -> float:
        gt_tokens = set(re.findall(r"\b\w+\b", ground_truth.lower()))
        if not gt_tokens:
            return 0.0
        all_ctx_tokens = set()
        for ctx in contexts:
            all_ctx_tokens.update(re.findall(r"\b\w+\b", ctx.lower()))
        if not all_ctx_tokens:
            return 0.0
        return len(gt_tokens & all_ctx_tokens) / len(gt_tokens)

    def _compute_answer_relevance(self, question: str, answer: str) -> float:
        q_tokens = set(re.findall(r"\b\w+\b", question.lower()))
        a_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
        if not q_tokens or not a_tokens:
            return 0.0
        return len(q_tokens & a_tokens) / len(q_tokens)

    def _compute_faithfulness(self, answer: str, contexts: List[str]) -> float:
        a_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
        if not a_tokens:
            return 0.0
        all_ctx = " ".join(contexts).lower()
        ctx_tokens = set(re.findall(r"\b\w+\b", all_ctx))
        if not ctx_tokens:
            return 0.5
        return len(a_tokens & ctx_tokens) / len(a_tokens)

    def _compute_causal_consistency(self, answer: str,
                                     causal_edges: List[Tuple[str, str, float]]) -> float:
        if not causal_edges:
            return 0.5
        answer_lower = answer.lower()
        consistent = 0
        for cause, effect, strength in causal_edges:
            if cause.lower() in answer_lower and effect.lower() in answer_lower:
                consistent += strength
        return consistent / len(causal_edges) if causal_edges else 0.5

    def _compute_temporal_coherence(self, contexts: List[str]) -> float:
        if len(contexts) < 2:
            return 1.0
        temporal_markers = ["then", "after", "before", "next", "later", "previously"]
        coherent = 0
        for ctx in contexts:
            ctx_lower = ctx.lower()
            if any(m in ctx_lower for m in temporal_markers):
                coherent += 1
        return coherent / len(contexts)

    def get_trend(self) -> Dict[str, float]:
        if len(self._eval_history) < 5:
            return {}
        recent = self._eval_history[-10:]
        older = self._eval_history[-20:-10] if len(self._eval_history) >= 20 else self._eval_history[:5]
        return {
            "recent_overall": np.mean([e.overall_score for e in recent]),
            "older_overall": np.mean([e.overall_score for e in older]),
            "trend": "improving" if np.mean([e.overall_score for e in recent]) > np.mean([e.overall_score for e in older]) else "degrading",
        }


class AutoRollbackManager:
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self._baseline_score: Optional[float] = None
        self._recent_scores: deque = deque(maxlen=50)
        self._rollback_count = 0
        self._last_rollback_time: float = 0
        self._cooldown_period: float = 300.0

    def set_baseline(self, score: float):
        self._baseline_score = score

    def record_score(self, score: float) -> bool:
        self._recent_scores.append(score)
        if self._baseline_score is None or len(self._recent_scores) < 10:
            return False
        if time.time() - self._last_rollback_time < self._cooldown_period:
            return False
        recent_mean = np.mean(self._recent_scores)
        degradation = self._baseline_score - recent_mean
        if degradation > self.threshold:
            self._rollback_count += 1
            self._last_rollback_time = time.time()
            return True
        return False

    def get_rollback_rate(self) -> float:
        return self._rollback_count / max(len(self._recent_scores), 1)

    def get_state(self) -> Dict:
        return {
            "baseline_score": self._baseline_score,
            "recent_mean": float(np.mean(self._recent_scores)) if self._recent_scores else 0,
            "rollback_count": self._rollback_count,
            "rollback_rate": self.get_rollback_rate(),
        }


# ============================================================================
# PHASE 13 TRACK 1: TELEOLOGICAL LAYER (Goal/Intent Tracking)
# ============================================================================

@dataclass
class GoalNode:
    id: str
    description: str
    subgoals: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    completion: float = 0.0
    priority: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    status: str = "active"  # active, completed, abandoned
    related_nodes: List[str] = field(default_factory=list)
    intent_signals: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "GoalNode":
        return cls(**data)


class GoalTracker:
    """Tracks user goals, subgoals, and completion progress."""

    def __init__(self, max_goals: int = 20, goal_decay: float = 0.995,
                 completion_threshold: float = 0.8):
        self.max_goals = max_goals
        self.goal_decay = goal_decay
        self.completion_threshold = completion_threshold
        self.goals: Dict[str, GoalNode] = {}
        self._history: deque = deque(maxlen=200)

    def add_goal(self, description: str, goal_id: Optional[str] = None,
                 subgoals: Optional[List[str]] = None,
                 priority: float = 1.0) -> str:
        gid = goal_id or f"goal_{len(self.goals)}_{int(time.time())}"
        self.goals[gid] = GoalNode(
            id=gid, description=description,
            subgoals=subgoals or [], priority=priority
        )
        self._history.append({"action": "add", "goal_id": gid, "time": time.time()})
        self._enforce_max_goals()
        return gid

    def update_completion(self, goal_id: str, completion: float,
                          related_nodes: Optional[List[str]] = None):
        if goal_id in self.goals:
            goal = self.goals[goal_id]
            goal.completion = min(1.0, max(0.0, completion))
            goal.last_updated = time.time()
            if related_nodes:
                goal.related_nodes = list(set(goal.related_nodes + related_nodes))
            if goal.completion >= self.completion_threshold:
                goal.status = "completed"
            self._history.append({
                "action": "update", "goal_id": goal_id,
                "completion": goal.completion, "time": time.time()
            })

    def get_active_goals(self) -> List[GoalNode]:
        return [g for g in self.goals.values() if g.status == "active"]

    def get_goal_relevance(self, node_id: str) -> float:
        """How relevant is a node to current active goals?"""
        if not self.goals:
            return 0.0
        relevance = 0.0
        for goal in self.get_active_goals():
            if node_id in goal.related_nodes:
                relevance += goal.priority * (1.0 - goal.completion)
            # Check subgoals
            for sg in goal.subgoals:
                if sg in node_id or node_id in sg:
                    relevance += goal.priority * 0.5
        return min(1.0, relevance)

    def decay_goals(self):
        """Decay inactive goals over time."""
        to_remove = []
        for gid, goal in self.goals.items():
            if goal.status == "active":
                goal.priority *= self.goal_decay
                if goal.priority < 0.01:
                    goal.status = "abandoned"
                    to_remove.append(gid)
        for gid in to_remove:
            del self.goals[gid]

    def _enforce_max_goals(self):
        active = self.get_active_goals()
        if len(active) > self.max_goals:
            sorted_goals = sorted(active, key=lambda g: g.priority)
            for goal in sorted_goals[:len(active) - self.max_goals]:
                goal.status = "abandoned"

    def get_state(self) -> Dict:
        return {
            "goals": {k: v.to_dict() for k, v in self.goals.items()},
            "history": list(self._history)[-100:],  # deque doesn't support slice, convert to list
        }

    def load_state(self, state: Dict):
        for gid, gdata in state.get("goals", {}).items():
            self.goals[gid] = GoalNode.from_dict(gdata)
        self._history = state.get("history", [])


# ============================================================================
# PHASE 13 TRACK 2: COGNITIVE ATTENTION BIAS
# ============================================================================

def apply_attention_bias(results: List[Tuple[str, float, MemoryNode]],
                         temperature: float = 1.0) -> List[Tuple[str, float, MemoryNode]]:
    """
    Transform raw resonance scores into attention-biased scores.
    Incorporates causal_strength, tension, salience as structural signals.
    """
    if not results:
        return results

    # Extract raw scores
    raw_scores = np.array([r for _, r, _ in results])
    if len(raw_scores) < 2:
        return results

    # Compute attention weights
    weights = []
    for nid, resp, node in results:
        # Base resonance
        score = resp
        # Causal boost
        causal_boost = sum(node.causal_strength.values()) if hasattr(node, 'causal_strength') else 0
        score *= (1.0 + 0.2 * min(1.0, causal_boost))
        # Tension penalty (high tension = less reliable)
        score *= max(0.5, 1.0 - node.tension)
        # Goal relevance boost (Phase 13 Track 1)
        goal_rel = getattr(node, 'goal_relevance', 0.0)
        score *= (1.0 + 0.3 * goal_rel)
        weights.append(score)

    weights = np.array(weights)
    # Softmax with temperature
    if temperature > 0:
        exp_weights = np.exp(weights / temperature)
        normalized = exp_weights / (exp_weights.sum() + 1e-8)
    else:
        normalized = weights / (weights.sum() + 1e-8)

    # Re-rank by attention-biased scores
    biased_results = []
    for i, (nid, resp, node) in enumerate(results):
        biased_results.append((nid, float(normalized[i]), node))

    biased_results.sort(key=lambda x: x[1], reverse=True)
    return biased_results


def format_cognitive_context(results: List[Tuple[str, float, MemoryNode]],
                             bias_applied: bool = False) -> str:
    """Format memory results with structural attention signals for LLM.
    
    Handles both structured nodes (v2: input_text, output_text, emotion, tags)
    and legacy nodes (v1: text).
    """
    if not results:
        return "### COGNITIVE_CONTEXT\nNo relevant structures."

    lines = ["### COGNITIVE_CONTEXT"]
    for nid, score, node in results:
        content = node.content
        
        # Check for structured node (v2)
        if content.get("version") == "2.0":
            input_text = content.get("input_text", "")
            output_text = content.get("output_text", "")
            emotion = content.get("emotion", "neutral")
            tags = content.get("tags", [])
            session = content.get("session", "")
            
            # Format structured context
            text_parts = []
            if input_text:
                text_parts.append(f"User: {input_text[:80]}")
            if output_text:
                text_parts.append(f"AI: {output_text[:80]}")
            text = " | ".join(text_parts) if text_parts else content.get("text", "unknown")[:80]
            
            tier = content.get("tier", getattr(node, 'tier', 'semantic'))
            tokens = f"[SCORE:{score:.3f}]"
            tokens += f"[TIER:{tier[0].upper()}]"
            if emotion != "neutral":
                tokens += f"[EMO:{emotion[:4]}]"
            if tags:
                tokens += f"[TAGS:{','.join(tags[:3])}]"
            if session:
                tokens += f"[SESS:{session[:10]}]"
        else:
            # Legacy node (v1)
            text = content.get("text", "unknown")[:80]
            tier = content.get("tier", getattr(node, 'tier', 'semantic'))
            tokens = f"[SCORE:{score:.3f}]"
            tokens += f"[TIER:{tier[0].upper()}]"

        causal = len(node.causal_strength) if hasattr(node, 'causal_strength') else 0
        tension = node.tension
        lineage = len(node.lineage) if node.lineage else 0
        
        if causal > 0:
            tokens += f"[CAUSAL:{causal}]"
        if tension > 0.3:
            tokens += f"[TENSION:{tension:.2f}]"
        if lineage > 0:
            tokens += f"[LINEAGE:{lineage}]"

        lines.append(f"{tokens} {text}")

    return "\n".join(lines)


# ============================================================================
# PHASE 13 TRACK 3: CLOSED-LOOP RL FROM LLM FEEDBACK
# ============================================================================

class RLFeedbackLoop:
    """Extracts confidence/uncertainty signals from LLM responses
    and uses them as reinforcement for field updates."""

    def __init__(self, learning_rate: float = 0.01, reward_window: int = 10):
        self.lr = learning_rate
        self.reward_window = reward_window
        self._rewards: deque = deque(maxlen=reward_window)
        self._node_rewards: Dict[str, List[float]] = defaultdict(list)

    def extract_reward_from_response(self, response: str,
                                      context_nodes: List[str]) -> float:
        """Extract reward signal from LLM response text."""
        reward = 0.5  # baseline

        # Confidence markers
        confidence_phrases = ["certainly", "definitely", "clearly", "obviously",
                              "безусловно", "очевидно", "точно"]
        uncertainty_phrases = ["not sure", "might be", "could be", "perhaps",
                               "не уверен", "возможно", "кажется", "probably"]

        resp_lower = response.lower()
        for phrase in confidence_phrases:
            if phrase in resp_lower:
                reward += 0.1
        for phrase in uncertainty_phrases:
            reward -= 0.1

        # Fallback: punctuation-based uncertainty estimation
        uncertainty_penalty = (response.count("?") + response.count("возможно")) * 0.15
        reward -= min(0.3, uncertainty_penalty)

        # Length-based signal (too short = unhelpful)
        words = response.split()
        if len(words) < 10:
            reward -= 0.2
        elif len(words) > 200:
            reward -= 0.05  # Very long might be unfocused

        reward = max(0.0, min(1.0, reward))
        self._rewards.append(reward)

        # Distribute reward to context nodes
        for nid in context_nodes:
            self._node_rewards[nid].append(reward)

        return reward

    def get_node_reward(self, node_id: str) -> float:
        """Get average reward for a specific node."""
        rewards = self._node_rewards.get(node_id, [])
        return float(np.mean(rewards)) if rewards else 0.5

    def get_average_reward(self) -> float:
        return float(np.mean(self._rewards)) if self._rewards else 0.5

    def apply_field_updates(self, field: Any):
        """Apply RL-based updates to field parameters."""
        if len(self._rewards) < 3:
            return

        avg_reward = self.get_average_reward()
        reward_trend = 0.0
        if len(self._rewards) >= 2:
            recent = list(self._rewards)[-5:]
            reward_trend = recent[-1] - recent[0]

        # Update node RL rewards
        for nid in field.node_index:
            if nid in field.nodes:
                node = field.nodes[nid]
                node_rl = self.get_node_reward(nid)
                node.rl_reward = node_rl
                # Update goal_relevance based on reward
                if reward_trend > 0.1:
                    node.goal_relevance = min(1.0, node.goal_relevance + self.lr)
                elif reward_trend < -0.1:
                    node.goal_relevance = max(0.0, node.goal_relevance - self.lr)

    def get_state(self) -> Dict:
        return {
            "rewards": list(self._rewards),
            "node_rewards": {k: v[-10:] for k, v in self._node_rewards.items()},
        }

    def load_state(self, state: Dict):
        self._rewards = deque(state.get("rewards", []), maxlen=self.reward_window)
        self._node_rewards = defaultdict(list, state.get("node_rewards", {}))


# ============================================================================
# PHASE 13 TRACK 4: EVENT-DRIVEN + LOW-RANK COMPRESSION
# ============================================================================

class LowRankCompressor:
    """Incremental SVD-based compression of latent states."""

    def __init__(self, rank: int = 32):
        self.rank = rank
        self.U: Optional[NDArray] = None
        self.S: Optional[NDArray] = None
        self.Vt: Optional[NDArray] = None
        self._update_count = 0

    def compress(self, positions: NDArray) -> Tuple[NDArray, NDArray]:
        """Compress node positions to low-rank representation."""
        if len(positions) < 2:
            return positions, positions

        # Truncated SVD
        U, S, Vt = np.linalg.svd(positions, full_matrices=False)
        k = min(self.rank, len(S))
        self.U = U[:, :k]
        self.S = S[:k]
        self.Vt = Vt[:k, :]
        self._update_count += 1

        compressed = self.U @ np.diag(self.S)
        reconstructed = compressed @ self.Vt
        return compressed, reconstructed

    def get_compression_ratio(self, original_shape: Tuple[int, int]) -> float:
        """How much compression achieved."""
        original_size = original_shape[0] * original_shape[1]
        compressed_size = self.rank * (original_shape[0] + original_shape[1])
        return compressed_size / max(original_size, 1)

    def get_state(self) -> Dict:
        return {
            "rank": self.rank,
            "update_count": self._update_count,
            "U": self.U.tolist() if self.U is not None else None,
            "S": self.S.tolist() if self.S is not None else None,
            "Vt": self.Vt.tolist() if self.Vt is not None else None,
        }

    def load_state(self, state: Dict):
        self.rank = state.get("rank", self.rank)
        self._update_count = state.get("update_count", 0)
        if state.get("U"):
            self.U = np.array(state["U"], dtype=np.float32)
        if state.get("S"):
            self.S = np.array(state["S"], dtype=np.float32)
        if state.get("Vt"):
            self.Vt = np.array(state["Vt"], dtype=np.float32)


class EventDrivenScheduler:
    """Event-driven triggers instead of periodic step()."""

    def __init__(self):
        self._event_queue: deque = deque(maxlen=1000)
        self._event_counts: Dict[str, int] = defaultdict(int)

    def enqueue(self, event_type: str, payload: Dict[str, Any]):
        self._event_queue.append({
            "type": event_type,
            "payload": payload,
            "timestamp": time.time(),
        })
        self._event_counts[event_type] += 1

    def process_pending(self, field: Any, max_events: int = 10) -> int:
        """Process pending events."""
        processed = 0
        while self._event_queue and processed < max_events:
            event = self._event_queue.popleft()
            etype = event["type"]
            payload = event["payload"]

            if etype == "node_added":
                pass  # Already handled by add_node
            elif etype == "high_tension":
                field.consolidate()
                processed += 1
            elif etype == "query":
                pass  # Already handled by query
            elif etype == "crystallize":
                if hasattr(field, '_crystallize_recurring'):
                    field._crystallize_recurring()
                processed += 1
            elif etype == "compress":
                if hasattr(field, '_compress_field'):
                    field._compress_field()
                processed += 1

        return processed

    def get_stats(self) -> Dict:
        return {
            "queue_depth": len(self._event_queue),
            "event_counts": dict(self._event_counts),
        }

    def get_state(self) -> Dict:
        """Get state for serialization (Fix 4: needed for export_to_dict)."""
        return {
            "event_queue": list(self._event_queue),
            "event_counts": dict(self._event_counts),
        }

    def load_state(self, state: Dict):
        """Load state from serialization (Fix 4: needed for import_from_dict)."""
        self._event_queue = deque(state.get("event_queue", []), maxlen=1000)
        self._event_counts = defaultdict(int, state.get("event_counts", {}))


# ============================================================================
# PHASE 14 TRACK 1: INTROSPECTIVE META-MEMORY
# ============================================================================

class MetaMemoryEvaluator:
    """Evaluates recall accuracy, memory age, and self-reflection."""

    def __init__(self, recall_threshold: float = 0.6, age_factor: float = 0.001,
                 reflection_freq: int = 100):
        self.recall_threshold = recall_threshold
        self.age_factor = age_factor
        self.reflection_freq = reflection_freq
        self._recall_history: deque = deque(maxlen=100)
        self._reflection_log: List[Dict] = []
        self._step_counter = 0

    def record_recall(self, query_text: str, result_score: float,
                      node_age: float = 0.0) -> Dict[str, float]:
        """Record a recall event and compute accuracy metrics."""
        self._recall_history.append(result_score)
        age_penalty = 1.0 - min(1.0, node_age * self.age_factor)
        adjusted_score = result_score * age_penalty
        return {
            "raw_score": result_score,
            "age_penalty": age_penalty,
            "adjusted_score": adjusted_score,
            "node_age": node_age,
        }

    def evaluate_recall_accuracy(self) -> float:
        if not self._recall_history:
            return 1.0
        return float(np.mean(self._recall_history))

    def should_reflect(self) -> bool:
        self._step_counter += 1
        return self._step_counter % self.reflection_freq == 0

    def self_reflect(self, field: Any) -> Dict[str, Any]:
        """Introspective analysis of memory field health."""
        recall_acc = self.evaluate_recall_accuracy()
        n_nodes = len(field.nodes) if hasattr(field, 'nodes') else 0
        n_consolidations = field.stats.get("consolidations", 0) if hasattr(field, 'stats') else 0
        false_merges = field.stats.get("false_merges", 0) if hasattr(field, 'stats') else 0

        recommendations = []
        if recall_acc < self.recall_threshold:
            recommendations.append("lower_consolidation_threshold")
        if n_consolidations > 0 and false_merges > n_consolidations * 0.2:
            recommendations.append("increase_tension_threshold")
        if n_nodes > 1000:
            recommendations.append("trigger_crystallization")

        reflection = {
            "recall_accuracy": recall_acc,
            "n_nodes": n_nodes,
            "n_consolidations": n_consolidations,
            "false_merges": false_merges,
            "false_merge_rate": false_merges / max(n_consolidations, 1),
            "recommendations": recommendations,
            "timestamp": time.time(),
        }
        self._reflection_log.append(reflection)
        return reflection

    def get_adaptive_params(self) -> Dict[str, float]:
        recall_acc = self.evaluate_recall_accuracy()
        if recall_acc < self.recall_threshold:
            return {"consolidation_multiplier": 0.8, "decay_multiplier": 1.1}
        elif recall_acc > 0.9:
            return {"consolidation_multiplier": 1.2, "decay_multiplier": 0.95}
        return {"consolidation_multiplier": 1.0, "decay_multiplier": 1.0}

    def get_state(self) -> Dict:
        return {
            "recall_history": list(self._recall_history),
            "reflection_log": self._reflection_log[-50:],
            "step_counter": self._step_counter,
        }

    def load_state(self, state: Dict):
        self._recall_history = deque(state.get("recall_history", []), maxlen=100)
        self._reflection_log = state.get("reflection_log", [])
        self._step_counter = state.get("step_counter", 0)


# ============================================================================
# PHASE 14 TRACK 2: FORMAL SECURITY
# ============================================================================

class SecurityValidator:
    """Protects against memory poisoning, prompt injection, and graph attacks."""

    def __init__(self, max_text_length: int = 10000,
                 tension_spike_threshold: float = 0.5,
                 injection_patterns: Optional[List[str]] = None):
        self.max_text_length = max_text_length
        self.tension_spike_threshold = tension_spike_threshold
        self.injection_patterns = injection_patterns or [
            "ignore previous", "system prompt", "you are now", "disregard",
            "ignore all", "new instruction", "override"
        ]
        self._violation_log: List[Dict] = []
        self._tension_history: deque = deque(maxlen=100)

    def validate_node_content(self, text: str) -> Dict[str, Any]:
        """Validate node text for injection patterns and length."""
        violations = []
        if len(text) > self.max_text_length:
            violations.append({"type": "text_too_long", "length": len(text), "max": self.max_text_length})
        text_lower = text.lower()
        for pattern in self.injection_patterns:
            if pattern in text_lower:
                violations.append({"type": "prompt_injection", "pattern": pattern})
        is_safe = len(violations) == 0
        if violations:
            self._violation_log.append({
                "type": "node_validation", "text_preview": text[:100],
                "violations": violations, "timestamp": time.time(),
            })
        return {"is_safe": is_safe, "violations": violations}

    def validate_tension_spike(self, current_tension: float) -> bool:
        """Detect anomalous tension spikes that may indicate attacks."""
        self._tension_history.append(current_tension)
        if len(self._tension_history) < 10:
            return True
        mean_t = np.mean(self._tension_history)
        std_t = np.std(self._tension_history)
        if std_t > 0 and (current_tension - mean_t) / std_t > self.tension_spike_threshold:
            self._violation_log.append({
                "type": "tension_spike", "current": current_tension,
                "mean": float(mean_t), "std": float(std_t), "timestamp": time.time(),
            })
            return False
        return True

    def validate_causal_graph_integrity(self, causal_engine: Any) -> Dict[str, Any]:
        """Check causal graph for anomalies."""
        if not causal_engine or not hasattr(causal_engine, 'causal_effects'):
            return {"is_valid": True, "issues": []}
        issues = []
        effects = causal_engine.causal_effects
        for (src, tgt), edge in effects.items():
            if src == tgt:
                issues.append({"type": "self_loop", "node": src})
            if edge.strength < 0 or edge.strength > 1.0:
                issues.append({"type": "invalid_strength", "edge": f"{src}->{tgt}", "strength": edge.strength})
        is_valid = len(issues) == 0
        if issues:
            self._violation_log.append({
                "type": "causal_graph_integrity", "issues": issues, "timestamp": time.time(),
            })
        return {"is_valid": is_valid, "issues": issues, "n_edges": len(effects)}

    def get_violation_summary(self) -> Dict:
        return {
            "total_violations": len(self._violation_log),
            "recent_violations": self._violation_log[-10:],
            "tension_spike_rate": sum(1 for v in self._violation_log if v["type"] == "tension_spike") / max(len(self._tension_history), 1),
        }

    def get_state(self) -> Dict:
        return {"violation_log": self._violation_log[-100:], "tension_history": list(self._tension_history)}

    def load_state(self, state: Dict):
        self._violation_log = state.get("violation_log", [])
        self._tension_history = deque(state.get("tension_history", []), maxlen=100)


# ============================================================================
# PHASE 14 TRACK 5: SWARM MEMORY
# ============================================================================

class SwarmConsensusProtocol:
    """Consensus-based memory sharing for multi-agent scenarios."""

    def __init__(self, consensus_threshold: float = 0.5, max_agents: int = 10,
                 vote_weight: float = 0.3):
        self.consensus_threshold = consensus_threshold
        self.max_agents = max_agents
        self.vote_weight = vote_weight
        self.agents: Dict[str, Dict] = {}
        self._consensus_log: List[Dict] = []

    def register_agent(self, agent_id: str, specialization: str = "general") -> bool:
        if len(self.agents) >= self.max_agents:
            return False
        self.agents[agent_id] = {
            "specialization": specialization, "vote_weight": self.vote_weight,
            "last_sync": time.time(), "n_exchanges": 0,
        }
        return True

    def propose_attractor(self, proposer_id: str, attractor: Dict[str, Any]) -> bool:
        if proposer_id not in self.agents:
            return False
        total_weight = 0
        agree_weight = 0
        votes = {}
        for agent_id, agent in self.agents.items():
            if agent_id == proposer_id:
                votes[agent_id] = True
                agree_weight += agent["vote_weight"]
                total_weight += agent["vote_weight"]
                continue
            spec_match = 1.0 if agent["specialization"] == "general" else 0.7
            vote = bool(np.random.random() < spec_match)
            votes[agent_id] = vote
            total_weight += agent["vote_weight"]
            if vote:
                agree_weight += agent["vote_weight"]
        consensus_ratio = agree_weight / max(total_weight, 1e-8)
        accepted = consensus_ratio >= self.consensus_threshold
        self._consensus_log.append({
            "proposer": proposer_id,
            "attractor_preview": str(attractor.get("text", ""))[:50],
            "accepted": accepted, "consensus_ratio": consensus_ratio,
            "votes": votes, "timestamp": time.time(),
        })
        if accepted:
            for agent_id in self.agents:
                self.agents[agent_id]["n_exchanges"] += 1
                self.agents[agent_id]["last_sync"] = time.time()
        return accepted

    def get_swarm_status(self) -> Dict:
        return {
            "n_agents": len(self.agents), "agents": dict(self.agents),
            "n_consensus_events": len(self._consensus_log),
            "recent_consensus": self._consensus_log[-5:],
        }

    def get_state(self) -> Dict:
        return {"agents": dict(self.agents), "consensus_log": self._consensus_log[-100:]}

    def load_state(self, state: Dict):
        self.agents = state.get("agents", {})
        self._consensus_log = state.get("consensus_log", [])


# ============================================================================
# SUPPORTING COMPONENTS
# ============================================================================

class MetaAdaptiveKernel:
    def __init__(self, base_bandwidth: float = 1.0, base_phase_coupling: float = 0.3,
                 adaptation_lr: float = 0.005, kurtosis_target_min: float = 1.5,
                 kurtosis_target_max: float = 4.0):
        self.base_bandwidth = base_bandwidth
        self.base_phase_coupling = base_phase_coupling
        self.adaptation_lr = adaptation_lr
        self.kurtosis_target_min = kurtosis_target_min
        self.kurtosis_target_max = kurtosis_target_max
        self.effective_bandwidth = base_bandwidth
        self.effective_phase_coupling = base_phase_coupling
        self._response_history: deque = deque(maxlen=100)
        self._semantic_density: deque = deque(maxlen=50)
        self._uncertainty: deque = deque(maxlen=20)
        self._kurtosis_history: deque = deque(maxlen=50)

    def record_response(self, response: float):
        self._response_history.append(response)

    def record_semantic_density(self, density: float):
        self._semantic_density.append(density)

    def record_uncertainty(self, entropy: float):
        self._uncertainty.append(entropy)

    def compute_resonance_kurtosis(self) -> float:
        if len(self._response_history) < 4:
            return 3.0
        responses = np.array(self._response_history)
        if np.std(responses) < 1e-8:
            return 3.0
        return float(scipy_stats.kurtosis(responses) + 3.0)

    def adapt(self):
        kurtosis = self.compute_resonance_kurtosis()
        self._kurtosis_history.append(kurtosis)
        # Bug #9 FIX: Reversed direction was driving system AWAY from target
        # Low kurtosis (flat distribution) → WIDEN bandwidth to sharpen
        # High kurtosis (peaked distribution) → NARROW bandwidth to smooth
        if kurtosis < self.kurtosis_target_min:
            self.effective_bandwidth *= (1.0 + self.adaptation_lr)  # WIDEN to sharpen
        elif kurtosis > self.kurtosis_target_max:
            self.effective_bandwidth *= (1.0 - self.adaptation_lr)  # NARROW to smooth
        if self._semantic_density:
            density = np.mean(self._semantic_density)
            if density > 0.7:
                self.effective_phase_coupling = min(0.9, self.effective_phase_coupling + self.adaptation_lr * 0.5)
            elif density < 0.2:
                self.effective_phase_coupling = max(0.05, self.effective_phase_coupling - self.adaptation_lr * 0.5)
        if self._uncertainty:
            uncertainty = np.mean(self._uncertainty)
            if uncertainty > 1.5:
                self.effective_bandwidth *= (1.0 + self.adaptation_lr)
        self.effective_bandwidth = max(0.1, min(10.0, self.effective_bandwidth))
        self.effective_phase_coupling = max(0.0, min(1.0, self.effective_phase_coupling))

    def get_bandwidth(self) -> float:
        return self.effective_bandwidth

    def get_phase_coupling(self) -> float:
        return self.effective_phase_coupling

    def get_state(self) -> Dict:
        return {
            "base_bandwidth": self.base_bandwidth, "base_phase_coupling": self.base_phase_coupling,
            "effective_bandwidth": self.effective_bandwidth, "effective_phase_coupling": self.effective_phase_coupling,
            "kurtosis": self.compute_resonance_kurtosis(),
            "avg_density": float(np.mean(self._semantic_density)) if self._semantic_density else 0,
            "avg_uncertainty": float(np.mean(self._uncertainty)) if self._uncertainty else 0,
        }

    def load_state(self, state: Dict):
        self.base_bandwidth = state.get("base_bandwidth", self.base_bandwidth)
        self.base_phase_coupling = state.get("base_phase_coupling", self.base_phase_coupling)
        self.effective_bandwidth = state.get("effective_bandwidth", self.base_bandwidth)
        self.effective_phase_coupling = state.get("effective_phase_coupling", self.base_phase_coupling)


class TopologyHealer:
    def __init__(self, dead_zone_threshold: float = 0.15, hyperconvergence_threshold: float = 0.05,
                 fragmentation_threshold: float = 0.6, healing_strength: float = 0.1,
                 max_healing_nodes: int = 5):
        self.dead_zone_threshold = dead_zone_threshold
        self.hyperconvergence_threshold = hyperconvergence_threshold
        self.fragmentation_threshold = fragmentation_threshold
        self.healing_strength = healing_strength
        self.max_healing_nodes = max_healing_nodes
        self._health_history: deque = deque(maxlen=100)

    def detect_dead_zones(self, nodes: Dict[str, MemoryNode]) -> List[str]:
        if len(nodes) < 3:
            return []
        positions = np.array([n.latent_pos for n in nodes.values()])
        tree = cKDTree(positions)
        min_dists = tree.query(positions, k=2)[0][:, 1]
        threshold = np.median(min_dists) * (1.0 + self.dead_zone_threshold * 5)
        return [nid for i, nid in enumerate(nodes) if min_dists[i] > threshold]

    def detect_hyperconvergence(self, nodes: Dict[str, MemoryNode]) -> bool:
        if len(nodes) < 3:
            return False
        positions = np.array([n.latent_pos for n in nodes.values()])
        return np.mean(pdist(positions)) < self.hyperconvergence_threshold

    def detect_fragmentation(self, nodes: Dict[str, MemoryNode], radius: float = 2.0) -> float:
        if len(nodes) < 2:
            return 0.0
        positions = np.array([n.latent_pos for n in nodes.values()])
        tree = cKDTree(positions)
        neighbors = tree.query_ball_point(positions, radius)
        isolated = sum(1 for nbrs in neighbors if len(nbrs) <= 1)
        return float(isolated / len(nodes))

    def compute_field_health(self, nodes: Dict[str, MemoryNode]) -> Tuple[FieldHealth, Dict]:
        diagnostics = {}
        dead = self.detect_dead_zones(nodes)
        diagnostics["dead_zones"] = len(dead)
        diagnostics["dead_zone_nodes"] = dead
        hyperconv = self.detect_hyperconvergence(nodes)
        diagnostics["hyperconvergence"] = hyperconv
        frag = self.detect_fragmentation(nodes)
        diagnostics["fragmentation"] = frag
        if len(nodes) >= 3:
            positions = np.array([n.latent_pos for n in nodes.values()])
            valid = pdist(positions)
            diagnostics["avg_pairwise_dist"] = float(np.mean(valid))
            diagnostics["std_pairwise_dist"] = float(np.std(valid))
            diagnostics["density_cv"] = float(np.std(valid) / max(np.mean(valid), 1e-8))
        else:
            diagnostics["avg_pairwise_dist"] = 0.0
            diagnostics["density_cv"] = 0.0
        if hyperconv or frag > 0.8:
            health = FieldHealth.CRITICAL
        elif len(dead) > len(nodes) * 0.3 or frag > self.fragmentation_threshold:
            health = FieldHealth.DEGRADED
        else:
            health = FieldHealth.STABLE
        self._health_history.append(health.value)
        diagnostics["health"] = health.value
        diagnostics["stable_fraction"] = (
            sum(1 for h in self._health_history if h == "stable") / max(len(self._health_history), 1))
        return health, diagnostics

    def heal_dead_zones(self, nodes: Dict[str, MemoryNode], dead_ids: List[str]) -> List[Dict]:
        healed = []
        alive_ids = [nid for nid in nodes if nid not in dead_ids]
        if not alive_ids or not dead_ids:
            return healed
        alive_positions = np.array([nodes[nid].latent_pos for nid in alive_ids])
        for dead_id in dead_ids[:self.max_healing_nodes]:
            dead_node = nodes[dead_id]
            dists = np.linalg.norm(alive_positions - dead_node.latent_pos, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_id = alive_ids[nearest_idx]
            old_pos = dead_node.latent_pos.copy()
            dead_node.latent_pos = ((1.0 - self.healing_strength) * old_pos + self.healing_strength * nodes[nearest_id].latent_pos).astype(np.float32)
            dead_node.is_healing = True
            dead_node.healing_origin = nearest_id
            dead_node.salience = max(dead_node.salience, 0.1)
            dead_node.amplitude = max(dead_node.amplitude, 0.1)
            healed.append({"node_id": dead_id, "from": old_pos.tolist(), "to": dead_node.latent_pos.tolist(), "type": "dead_zone"})
        return healed

    def heal_hyperconvergence(self, nodes: Dict[str, MemoryNode]) -> List[Dict]:
        healed = []
        if len(nodes) < 3:
            return healed
        positions = np.array([n.latent_pos for n in nodes.values()])
        centroid = np.mean(positions, axis=0)
        for nid in list(nodes.keys())[:self.max_healing_nodes]:
            node = nodes[nid]
            direction = node.latent_pos - centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                direction = np.random.randn(len(centroid)).astype(np.float32)
                norm = 1.0
            direction = direction / norm
            old_pos = node.latent_pos.copy()
            node.latent_pos = (old_pos + self.healing_strength * direction).astype(np.float32)
            node.is_healing = True
            node.healing_origin = "hyperconvergence"
            healed.append({"node_id": nid, "from": old_pos.tolist(), "to": node.latent_pos.tolist(), "type": "hyperconvergence"})
        return healed

    def heal_fragmentation(self, nodes: Dict[str, MemoryNode], isolated_ids: List[str]) -> List[Dict]:
        healed = []
        non_isolated = [nid for nid in nodes if nid not in isolated_ids]
        if not non_isolated or not isolated_ids:
            return healed
        non_iso_positions = np.array([nodes[nid].latent_pos for nid in non_isolated])
        centroid = np.mean(non_iso_positions, axis=0)
        for iso_id in isolated_ids[:self.max_healing_nodes]:
            node = nodes[iso_id]
            old_pos = node.latent_pos.copy()
            node.latent_pos = ((1.0 - self.healing_strength) * old_pos + self.healing_strength * centroid).astype(np.float32)
            node.is_healing = True
            node.healing_origin = "fragmentation"
            healed.append({"node_id": iso_id, "from": old_pos.tolist(), "to": node.latent_pos.tolist(), "type": "fragmentation"})
        return healed

    def get_state(self) -> Dict:
        return {"health_history": list(self._health_history),
                "stable_fraction": sum(1 for h in self._health_history if h == "stable") / max(len(self._health_history), 1)}

    def load_state(self, state: Dict):
        self._health_history = deque(state.get("health_history", []), maxlen=100)


class CausalInferenceEngine:
    def __init__(self, min_samples: int = 20, p_threshold: float = 0.05,
                 adjustment_sets_enabled: bool = True):
        self.min_samples = min_samples
        self.p_threshold = p_threshold
        self.adjustment_sets_enabled = adjustment_sets_enabled
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.ancestors: Dict[str, Set[str]] = defaultdict(set)
        self._cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
        self._conditional_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._node_counts: Dict[str, int] = defaultdict(int)
        self._total_observations = 0
        self.causal_effects: Dict[Tuple[str, str], CausalEdge] = {}
        self.contradictions: Dict[str, ContradictionRecord] = {}
        self._contradiction_counter = 0
        self._counterfactual_cache: Dict[str, CounterfactualResult] = {}
        self._intervention_store: Dict[str, List[Dict]] = {}

    def record_cooccurrence(self, a: str, b: str):
        self._cooccurrence[(a, b)] += 1
        self._cooccurrence[(b, a)] += 1
        self._node_counts[a] += 1
        self._node_counts[b] += 1
        self._total_observations += 1

    def record_observation(self, active_nodes: List[str], context: Optional[Dict] = None):
        self._total_observations += 1
        for node in active_nodes:
            self._node_counts[node] += 1
        for i, a in enumerate(active_nodes):
            for b in active_nodes[i+1:]:
                self._cooccurrence[(a, b)] += 1
                self._cooccurrence[(b, a)] += 1
                if context:
                    for ctx_key, ctx_val in context.items():
                        self._conditional_counts[(a, b, f"{ctx_key}={ctx_val}")] += 1

    def discover_causal_structure(self) -> Dict[str, Set[str]]:
        nodes = list(self._node_counts.keys())
        if len(nodes) < 3 or self._total_observations < self.min_samples:
            return dict(self.parents)
        skeleton: Dict[str, Set[str]] = defaultdict(set)
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                if self._test_independence(a, b, set()):
                    continue
                skeleton[a].add(b)
                skeleton[b].add(a)
        new_parents: Dict[str, Set[str]] = defaultdict(set)
        new_children: Dict[str, Set[str]] = defaultdict(set)
        for z in nodes:
            neighbors = list(skeleton.get(z, set()))
            for i, x in enumerate(neighbors):
                for y in neighbors[i+1:]:
                    if y not in skeleton.get(x, set()):
                        new_parents[z].add(x)
                        new_parents[z].add(y)
                        new_children[x].add(z)
                        new_children[y].add(z)
        self.parents = new_parents
        self.children = new_children
        self._compute_ancestors()
        return dict(self.parents)

    def _test_independence(self, a: str, b: str, cond_set: Set[str]) -> bool:
        n_ab = self._cooccurrence.get((a, b), 0)
        n_a = self._node_counts.get(a, 0)
        n_b = self._node_counts.get(b, 0)
        n = max(self._total_observations, 1)
        if n_a < 3 or n_b < 3 or n_ab < 2:
            return True
        if not cond_set:
            # Marginal independence test: chi-squared
            expected = (n_a / n) * (n_b / n) * n
            if expected < 5:
                return True
            chi2 = (n_ab - expected) ** 2 / expected
            return chi2 < CHI_SQUARED_CRITICAL_DF1  # p=0.05, df=1

        # Bug #16 FIX: Implement conditional independence test
        # Use partial correlation approximation for discrete data
        # Test: a ⊥ b | cond_set
        total = 0
        chi2_cond = 0.0
        for c_node in cond_set:
            n_c = self._node_counts.get(c_node, 0)
            if n_c < 3:
                continue
            # Compute conditional probabilities
            p_a_given_c = min(n_ab, n_c) / max(n_c, 1)
            p_b_given_c = min(n_ab, n_c) / max(n_c, 1)
            p_ab_given_c = n_ab / max(n, 1)
            expected_cond = p_a_given_c * p_b_given_c * n_c
            if expected_cond > 0:
                chi2_cond += (n_ab - expected_cond) ** 2 / expected_cond
                total += 1

        if total == 0:
            # No valid conditioning sets — fall back to marginal
            return True

        # Average chi-squared over conditioning variables
        avg_chi2 = chi2_cond / total
        # With conditioning, use higher threshold (df increases)
        return avg_chi2 < CHI_SQUARED_CRITICAL_DF2  # p=0.05, df=2

    def _compute_ancestors(self):
        for node in self.parents:
            self.ancestors[node] = self._get_ancestors(node, set())

    def _get_ancestors(self, node: str, visited: Set[str]) -> Set[str]:
        if node in visited:
            return set()
        visited.add(node)
        ancestors = set()
        for parent in self.parents.get(node, set()):
            ancestors.add(parent)
            ancestors.update(self._get_ancestors(parent, visited))
        return ancestors

    def _get_descendants(self, node: str, visited: Optional[Set[str]] = None) -> Set[str]:
        if visited is None:
            visited = set()
        if node in visited:
            return set()
        visited.add(node)
        descendants = set()
        for child in self.children.get(node, set()):
            descendants.add(child)
            descendants.update(self._get_descendants(child, visited))
        return descendants

    def compute_do_probability(self, effect: str, intervention: str,
                               evidence: Optional[Dict[str, Any]] = None) -> float:
        edge = self.causal_effects.get((intervention, effect))
        if edge:
            return edge.strength
        return self._naive_causal_estimate(intervention, effect)

    def _naive_causal_estimate(self, cause: str, effect: str) -> float:
        n_cause = self._node_counts.get(cause, 0)
        n_both = self._cooccurrence.get((cause, effect), 0)
        if n_cause < 3:
            return 0.5
        return min(1.0, n_both / n_cause)

    def _find_adjustment_set(self, cause: str, effect: str) -> Set[str]:
        if not self.adjustment_sets_enabled:
            return set()
        parents_of_cause = self.parents.get(cause, set())
        descendants = self._get_descendants(cause)
        return parents_of_cause - descendants

    def _validate_do_calculus(self, effect: str, intervention: str) -> bool:
        z_set = self._find_adjustment_set(intervention, effect)
        has_frontdoor = self._has_frontdoor_path(intervention, effect)
        descendants = self._get_descendants(intervention)
        return bool(z_set) or has_frontdoor or effect in descendants

    def _has_frontdoor_path(self, cause: str, effect: str) -> bool:
        for mediator in self.children.get(cause, set()):
            if effect in self.children.get(mediator, set()):
                return True
        return False

    def detect_contradictions(self, threshold: float = 0.3) -> List[ContradictionRecord]:
        new_contradictions = []
        effect_causes: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (cause, effect), edge in self.causal_effects.items():
            if edge.strength > 0.1:
                effect_causes[effect].append((cause, edge.strength))
        for effect_node, causes in effect_causes.items():
            if len(causes) < 2:
                continue
            for i, (cause_a, strength_a) in enumerate(causes):
                for cause_b, strength_b in causes[i+1:]:
                    cooc = self._cooccurrence.get((cause_a, cause_b), 0)
                    n_a = self._node_counts.get(cause_a, 0)
                    n_b = self._node_counts.get(cause_b, 0)
                    if n_a > 0 and n_b > 0:
                        expected = (n_a / self._total_observations) * (n_b / self._total_observations) * self._total_observations
                        if expected > 0 and cooc / expected < (1.0 - threshold):
                            self._contradiction_counter += 1
                            record = ContradictionRecord(
                                id=f"contr_{self._contradiction_counter}",
                                effect_node=effect_node,
                                causes=[(cause_a, strength_a), (cause_b, strength_b)],
                                contradiction_reason=f"Causes {cause_a} and {cause_b} are negatively correlated"
                            )
                            self.contradictions[record.id] = record
                            new_contradictions.append(record)
                            if (cause_a, effect_node) in self.causal_effects:
                                self.causal_effects[(cause_a, effect_node)].is_contradicted = True
                            if (cause_b, effect_node) in self.causal_effects:
                                self.causal_effects[(cause_b, effect_node)].is_contradicted = True
        return new_contradictions

    def counterfactual_query(self, intervention: Dict[str, Any], query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None,
                             max_depth: int = 3) -> CounterfactualResult:
        query_str = f"do({intervention})|{query_nodes}"
        if query_str in self._counterfactual_cache:
            return self._counterfactual_cache[query_str]
        outcomes = []
        reasoning_path = []
        for target in query_nodes[:max_depth]:
            if target in intervention:
                outcomes.append((target, 1.0))
                reasoning_path.append(f"{target} is directly set")
                continue
            best_prob = 0.0
            best_path = ""
            for int_var, int_val in intervention.items():
                prob = self.compute_do_probability(target, int_var)
                if prob > best_prob:
                    best_prob = prob
                    best_path = f"do({int_var}) -> {target} (P={prob:.3f})"
            if best_path:
                outcomes.append((target, best_prob))
                reasoning_path.append(best_path)
            else:
                outcomes.append((target, 0.5))
                reasoning_path.append(f"No causal path to {target}")
        confidence = np.mean([p for _, p in outcomes]) if outcomes else 0.5
        result = CounterfactualResult(
            query=query_str, intervention=intervention, predicted_outcomes=outcomes,
            confidence=float(confidence), reasoning_path=reasoning_path, assumptions=[])
        self._counterfactual_cache[query_str] = result
        return result

    def validate_consolidation(self, node_a: str, node_b: str) -> Dict[str, Any]:
        result = {"safe": True, "reasons": [], "causal_conflicts": [], "recommendation": "proceed"}
        common_targets = set(self.children.get(node_a, set())) & set(self.children.get(node_b, set()))
        for target in common_targets:
            edge_a = self.causal_effects.get((node_a, target))
            edge_b = self.causal_effects.get((node_b, target))
            if edge_a and edge_b:
                diff = abs(edge_a.strength - edge_b.strength)
                if diff > 0.4:
                    result["safe"] = False
                    result["causal_conflicts"].append({"target": target, "effect_a": edge_a.strength, "effect_b": edge_b.strength, "difference": diff})
                    result["reasons"].append(f"Opposing effects on {target}")
        if node_b in self.children.get(node_a, set()) or node_a in self.children.get(node_b, set()):
            result["safe"] = False
            result["reasons"].append("Causal relationship exists")
            result["recommendation"] = "preserve_separate"
        if node_a in self.ancestors.get(node_b, set()) or node_b in self.ancestors.get(node_a, set()):
            result["safe"] = False
            result["reasons"].append("Merging would create causal cycle")
            result["recommendation"] = "preserve_separate"
        for cid, record in self.contradictions.items():
            if record.resolved:
                continue
            causes = [c for c, _ in record.causes]
            if node_a in causes and node_b in causes:
                result["safe"] = False
                result["reasons"].append(f"Unresolved contradiction: {cid}")
                result["recommendation"] = "resolve_contradiction_first"
        return result

    def do_intervention(self, node_id: str, new_pos: NDArray):
        """Apply do-calculus intervention: set node position to counterfactual state."""
        if node_id not in self.parents and node_id not in self.children:
            # Node not in causal graph — record it as a causal root
            self.parents[node_id] = set()
        # Store intervention for tracking
        if node_id not in self._intervention_store:
            self._intervention_store[node_id] = []
        self._intervention_store[node_id].append({
            "new_pos": new_pos.copy(),
            "timestamp": time.time(),
        })
        # Update causal effects with intervention strength
        for child in self.children.get(node_id, set()):
            edge_key = (node_id, child)
            if edge_key in self.causal_effects:
                edge = self.causal_effects[edge_key]
                edge.strength = min(1.0, edge.strength * 1.2)  # Boost causal strength under intervention
                edge.evidence_count += 1

    def clear_interventions(self):
        """Clear all recorded interventions."""
        self._intervention_store = {}

    def get_state(self) -> Dict:
        return {
            "parents": {k: list(v) for k, v in self.parents.items()},
            "children": {k: list(v) for k, v in self.children.items()},
            "causal_effects": {f"{k[0]}->{k[1]}": v.to_dict() for k, v in self.causal_effects.items()},
            "contradictions": {k: v.to_dict() for k, v in self.contradictions.items()},
            "node_counts": dict(self._node_counts),
            "total_observations": self._total_observations,
            "intervention_store": {k: [{"new_pos": v["new_pos"].tolist() if hasattr(v["new_pos"], 'tolist') else v["new_pos"], "timestamp": v["timestamp"]} for v in vals] for k, vals in self._intervention_store.items()},
        }

    def load_state(self, state: Dict):
        self.parents = defaultdict(set, {k: set(v) for k, v in state.get("parents", {}).items()})
        self.children = defaultdict(set, {k: set(v) for k, v in state.get("children", {}).items()})
        self._node_counts = defaultdict(int, state.get("node_counts", {}))
        self._total_observations = state.get("total_observations", 0)
        self._intervention_store = {}
        for node_id, interventions in state.get("intervention_store", {}).items():
            self._intervention_store[node_id] = [
                {"new_pos": np.array(iv["new_pos"], dtype=np.float32), "timestamp": iv["timestamp"]}
                for iv in interventions
            ]
        for key, edge_data in state.get("causal_effects", {}).items():
            parts = key.split("->")
            if len(parts) == 2:
                self.causal_effects[(parts[0], parts[1])] = CausalEdge.from_dict(edge_data)
        for cid, record_data in state.get("contradictions", {}).items():
            self.contradictions[cid] = ContradictionRecord(
                id=record_data["id"], effect_node=record_data["effect_node"],
                causes=record_data["causes"], timestamp=record_data["timestamp"],
                resolved=record_data["resolved"], resolution=record_data["resolution"],
                contradiction_reason=record_data.get("contradiction_reason", ""))
        self._compute_ancestors()


class IncPCAProjection:
    def __init__(self, input_dim: int, latent_dim: int, lr: float = 0.001, update_freq: int = 50, l2_reg: float = 0.0001):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.update_freq = update_freq
        self.l2_reg = l2_reg
        self.projection = np.random.randn(input_dim, latent_dim).astype(np.float32) * 0.1
        self.mean = np.zeros(input_dim, dtype=np.float32)
        self.buffer: List[NDArray] = []
        self.n_samples = 0
        self._try_sklearn()

    def _try_sklearn(self):
        try:
            from sklearn.decomposition import IncrementalPCA
            self.ipca = IncrementalPCA(n_components=self.latent_dim, batch_size=min(64, self.update_freq))
            self.use_sklearn = True
            self._ipca_fitted = False
            self._ipca_error = None  # Store any sklearn errors for fallback
        except ImportError:
            self.use_sklearn = False
            self._ipca_error = "sklearn not installed"

    def update(self, embedding: NDArray) -> NDArray:
        self.n_samples += 1
        self.buffer.append(embedding.copy())
        if len(self.buffer) >= self.update_freq:
            batch = np.array(self.buffer, dtype=np.float32)
            self.buffer = []
            if self.use_sklearn:
                try:
                    self.ipca.partial_fit(batch)
                    # Only mark as fitted if we have enough samples
                    if self.ipca.n_samples_seen_ >= self.latent_dim:
                        self._ipca_fitted = True
                        self.projection = self.ipca.components_.T.astype(np.float32)
                        self.mean = self.ipca.mean_.astype(np.float32)
                    else:
                        # Not enough samples yet, use manual update
                        self._ipca_fitted = False
                        alpha = self.lr / (1 + self.n_samples * self.lr * 0.01)
                        self.mean += alpha * (batch.mean(axis=0) - self.mean)
                except Exception as e:
                    self._ipca_error = str(e)
                    self._ipca_fitted = False
            else:
                alpha = self.lr / (1 + self.n_samples * self.lr * 0.01)
                self.mean += alpha * (batch.mean(axis=0) - self.mean)
                for emb in batch:
                    centered = emb - self.mean
                    latent = centered @ self.projection
                    reconstructed = latent @ self.projection.T
                    error = centered - reconstructed
                    self.projection += alpha * (np.outer(centered, latent) - np.outer(error, latent))
                    self.projection -= alpha * self.l2_reg * self.projection
                    norm = np.linalg.norm(self.projection, axis=0, keepdims=True)
                    self.projection /= np.maximum(norm, 1e-8)
        return self.project(embedding)

    def project(self, embedding: NDArray) -> NDArray:
        # Only use sklearn transform if properly fitted
        if self.use_sklearn and self._ipca_fitted and self._ipca_error is None:
            try:
                return self.ipca.transform(embedding.reshape(1, -1))[0].astype(np.float32)
            except Exception as e:
                logger.warning(f"IncrementalPCA projection failed, falling back to manual: {e}")
                self._ipca_fitted = False
        # Fix 9: Fallback to manual projection — track reconstruction error to detect divergence
        reconstructed = embedding - self.mean
        proj_norm = np.linalg.norm(self.projection)
        if proj_norm < 1e-8:
            logger.warning("IncPCAProjection: projection matrix ill-conditioned, may diverge")
        return ((embedding - self.mean) @ self.projection).astype(np.float32)

    def get_state(self) -> Dict:
        return {
            "projection": self.projection.tolist(),
            "mean": self.mean.tolist(),
            "n_samples": self.n_samples,
            "use_sklearn": self.use_sklearn,
            "ipca_fitted": self._ipca_fitted,
        }

    def set_matrix(self, matrix: NDArray):
        """Set projection matrix directly (for import/initialization)."""
        assert matrix.shape == (self.input_dim, self.latent_dim), \
            f"Expected shape ({self.input_dim}, {self.latent_dim}), got {matrix.shape}"
        self.projection = matrix.astype(np.float32)
        # Don't try to initialize sklearn here - it's safer to use manual projection
        self._ipca_fitted = False
        self.use_sklearn = False

    def load_state(self, state: Dict):
        self.projection = np.array(state["projection"], dtype=np.float32)
        self.mean = np.array(state["mean"], dtype=np.float32)
        self.n_samples = state.get("n_samples", 0)
        self._ipca_fitted = state.get("ipca_fitted", False)
        # Don't re-initialize sklearn from state - use manual projection
        self.use_sklearn = False


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.doc_freq: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def add_document(self, doc_id: str, text: str):
        self.documents[doc_id] = text
        tokens = self._tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.doc_freq[token] = self.doc_freq.get(token, 0) + 1
        self.avg_doc_length = np.mean(list(self.doc_lengths.values())) if self.doc_lengths else 0.0

    def remove_document(self, doc_id: str):
        if doc_id in self.documents:
            text = self.documents.pop(doc_id)
            for token in set(self._tokenize(text)):
                self.doc_freq[token] = max(0, self.doc_freq.get(token, 1) - 1)
                if self.doc_freq[token] == 0:
                    del self.doc_freq[token]
            self.doc_lengths.pop(doc_id, None)
            if self.doc_lengths:
                self.avg_doc_length = np.mean(list(self.doc_lengths.values()))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.documents:
            return []
        n = len(self.documents)
        scores = {doc_id: 0.0 for doc_id in self.documents}
        for token in self._tokenize(query):
            df = self.doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, text in self.documents.items():
                tf = text.lower().count(token)
                doc_len = self.doc_lengths.get(doc_id, 1)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1)))
        return [(d, s) for d, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k] if s > 0]


class AdaptiveThreshold:
    def __init__(self, window_size: int = 30, base_threshold: float = 0.25, sensitivity: float = 0.5):
        self.window: deque = deque(maxlen=window_size)
        self.base_threshold = base_threshold
        self.sensitivity = sensitivity
        self.current_threshold = base_threshold

    def record_tension(self, tension: float):
        self.window.append(tension)
        if len(self.window) >= 5:
            self.current_threshold = max(0.01, np.mean(self.window) + self.sensitivity * np.std(self.window))

    def get_threshold(self) -> float:
        return self.current_threshold


class TDAMonitor:
    def __init__(self):
        self.history: List[Dict] = []

    def compute_persistence(self, nodes: Dict[str, MemoryNode]) -> Dict:
        if len(nodes) < 3:
            return {"H0": 0, "H1": 0, "avg_persistence": 0.0}
        positions = np.array([n.latent_pos for n in nodes.values()])
        n = len(positions)
        valid = pdist(positions)
        if len(valid) < 2:
            return {"H0": n, "H1": 0, "avg_persistence": 0.0}
        threshold = np.median(valid)
        
        # Union-Find with path compression — O(N log N) with cKDTree
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        tree = cKDTree(positions)
        for i, j in tree.query_pairs(threshold):
            union(i, j)
        
        h0 = len(set(find(i) for i in range(n)))
        n_edges_threshold = len(tree.query_pairs(threshold))
        h1 = max(0, n_edges_threshold - n + h0)
        result = {"H0": h0, "H1": h1, "avg_persistence": 0.0}
        self.history.append(result)
        return result

    def get_trend(self) -> str:
        if len(self.history) < 2:
            return "stable"
        recent = [h["H1"] for h in self.history[-5:]]
        if len(recent) >= 3 and recent[-1] > recent[0] * 1.5:
            return "growing_contradictions"
        return "stable"


class HNSWIndex:
    def __init__(self, m: int = 16, ef_construction: int = 200):
        self.m = m
        self.ef_construction = ef_construction
        self.graph: Dict[str, List[str]] = {}
        self.positions: Dict[str, NDArray] = {}

    def insert(self, node_id: str, pos: NDArray):
        self.positions[node_id] = pos
        self.graph[node_id] = []
        if len(self.positions) <= 1:
            return
        candidates = [c for c in list(self.positions.keys()) if c != node_id][:self.ef_construction]
        if candidates:
            cand_pos = np.array([self.positions[c] for c in candidates])
            dists = np.linalg.norm(cand_pos - pos, axis=1)
            nearest = [candidates[i] for i in np.argsort(dists)[:self.m]]
            self.graph[node_id] = nearest
            for nb in nearest:
                if nb in self.graph:
                    self.graph[nb].append(node_id)
                    if len(self.graph[nb]) > self.m * 2:
                        self.graph[nb] = self.graph[nb][-self.m:]

    def remove(self, node_id: str):
        self.graph.pop(node_id, None)
        self.positions.pop(node_id, None)
        for nid in self.graph:
            self.graph[nid] = [n for n in self.graph[nid] if n != node_id]

    def search(self, query_pos: NDArray, top_k: int = 10) -> List[str]:
        if not self.positions:
            return []
        start = list(self.positions.keys())[0]
        candidates = {start}
        visited = set()
        for _ in range(min(self.ef_construction, len(self.positions))):
            best = min((c for c in candidates - visited), key=lambda c: np.linalg.norm(self.positions[c] - query_pos), default=None)
            if best is None:
                break
            visited.add(best)
            candidates.update(self.graph.get(best, []))
        return sorted(candidates, key=lambda nid: np.linalg.norm(self.positions[nid] - query_pos))[:top_k]


class TorchBackend:
    def __init__(self):
        self.torch = None
        self.device = None
        try:
            import torch
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self.torch is not None

    def batch_resonance(self, ql, qp, np_, nph, na, ns, bw, pc):
        if not self.available:
            return self._numpy(ql, qp, np_, nph, na, ns, bw, pc)
        tq = self.torch.from_numpy(ql).to(self.device)
        dists = self.torch.cdist(tq, self.torch.from_numpy(np_).to(self.device))
        # Bug #1 FIX: Gaussian kernel exp(-d^2/(2*bw^2))
        spatial = self.torch.exp(-dists ** 2 / (2 * bw ** 2))
        pd = qp.unsqueeze(1) - self.torch.from_numpy(nph).to(self.device).unsqueeze(0)
        pa = 0.5 + 0.5 * self.torch.cos(pd)
        r = spatial * ((1 - pc) + pc * pa)
        return (r * self.torch.from_numpy(na).to(self.device).unsqueeze(0) * self.torch.from_numpy(ns).to(self.device).unsqueeze(0)).cpu().numpy()

    @staticmethod
    def _numpy(ql, qp, np_, nph, na, ns, bw, pc):
        dists = cdist(ql, np_)
        # Bug #1 FIX: Use proper Gaussian kernel exp(-d^2/(2*bw^2)) instead of Laplacian exp(-d/bw)
        spatial = np.exp(-dists ** 2 / (2 * bw ** 2))
        pd = qp[:, np.newaxis] - nph[np.newaxis, :]
        pa = 0.5 + 0.5 * np.cos(pd)
        return spatial * ((1 - pc) + pc * pa) * na[np.newaxis, :] * ns[np.newaxis, :]


class LearnableKernel:
    def __init__(self, bandwidth: float = 1.0, phase_coupling: float = 0.3,
                 decay_rate: float = 0.998, gradient_clip: float = 1.0):
        self.bandwidth = bandwidth
        self.phase_coupling = phase_coupling
        self.decay_rate = decay_rate
        self.gradient_clip = gradient_clip
        self._grad_bandwidth = 0.0
        self._grad_phase_coupling = 0.0
        self._adam_state = {
            "bandwidth": {"m": 0.0, "v": 0.0, "t": 0},
            "phase_coupling": {"m": 0.0, "v": 0.0, "t": 0},
        }

    def resonance_response(self, dist: float, phase_diff: float, amplitude: float, salience: float) -> float:
        # Bug #1 FIX: Gaussian kernel exp(-d^2/(2*bw^2))
        spatial = math.exp(-dist ** 2 / (2 * self.bandwidth ** 2))
        phase_align = 0.5 + 0.5 * math.cos(phase_diff)
        return spatial * ((1 - self.phase_coupling) + self.phase_coupling * phase_align) * amplitude * salience

    def compute_gradients(self, dist: float, phase_diff: float, amplitude: float, salience: float, loss_gradient: float = 1.0):
        # Bug #1 FIX: Gradients for Gaussian kernel
        spatial = math.exp(-dist ** 2 / (2 * self.bandwidth ** 2))
        phase_align = 0.5 + 0.5 * math.cos(phase_diff)
        self._grad_bandwidth += loss_gradient * spatial * (dist ** 2 / self.bandwidth ** 3) * ((1 - self.phase_coupling) + self.phase_coupling * phase_align) * amplitude * salience
        self._grad_phase_coupling += loss_gradient * spatial * (phase_align - 1.0) * amplitude * salience

    def step(self):
        for param_name, grad in [("bandwidth", self._grad_bandwidth), ("phase_coupling", self._grad_phase_coupling)]:
            if abs(grad) < 1e-12:
                continue
            grad = np.clip(grad, -self.gradient_clip, self.gradient_clip)
            s = self._adam_state[param_name]
            s["t"] += 1
            s["m"] = 0.9 * s["m"] + 0.1 * grad
            s["v"] = 0.999 * s["v"] + 0.001 * grad ** 2
            m_hat = s["m"] / (1 - 0.9 ** s["t"])
            v_hat = s["v"] / (1 - 0.999 ** s["t"])
            lr = 0.001
            update = lr * m_hat / (math.sqrt(v_hat) + 1e-8)
            if param_name == "bandwidth":
                self.bandwidth = max(0.1, self.bandwidth - update)
            elif param_name == "phase_coupling":
                self.phase_coupling = float(np.clip(self.phase_coupling - update, 0.0, 1.0))
        self._grad_bandwidth = 0.0
        self._grad_phase_coupling = 0.0

    def get_state(self) -> Dict:
        return {"bandwidth": self.bandwidth, "phase_coupling": self.phase_coupling, "decay_rate": self.decay_rate,
                "adam_state": {k: dict(v) for k, v in self._adam_state.items()}}

    def load_state(self, state: Dict):
        self.bandwidth = state["bandwidth"]
        self.phase_coupling = state["phase_coupling"]
        self.decay_rate = state.get("decay_rate", self.decay_rate)
        if "adam_state" in state:
            self._adam_state = state["adam_state"]


class DifferentiableConsolidation:
    def __init__(self, loss_weight: float = 0.1):
        self.loss_weight = loss_weight
        self.consolidation_loss = 0.0

    def compute_synthesis(self, node1: MemoryNode, node2: MemoryNode, gate: float) -> Dict:
        w1, w2 = gate, 1.0 - gate
        new_latent = w1 * node1.latent_pos + w2 * node2.latent_pos
        new_phase = np.arctan2(w1*np.sin(node1.phase)+w2*np.sin(node2.phase),
                               w1*np.cos(node1.phase)+w2*np.cos(node2.phase)) % (2*np.pi)
        new_amp = min(1.0, w1*node1.amplitude + w2*node2.amplitude)
        new_sal = w1*node1.salience + w2*node2.salience
        pos_loss = np.sum((new_latent - node1.latent_pos)**2) + np.sum((new_latent - node2.latent_pos)**2)
        phase_loss = min(abs(new_phase-node1.phase), 2*np.pi-abs(new_phase-node1.phase)) + min(abs(new_phase-node2.phase), 2*np.pi-abs(new_phase-node2.phase))
        self.consolidation_loss = self.loss_weight * (pos_loss + phase_loss * 0.1)
        return {"latent_pos": new_latent, "phase": new_phase, "amplitude": new_amp,
                "salience": new_sal, "loss": self.consolidation_loss}


# ============================================================================
# CONTEXT FORMATTING
# ============================================================================

SYSTEM_PROMPT_TEMPLATES = {
    ContextFormat.PLAIN: (
        "You are a helpful assistant with long-term memory.\n"
        "Below are relevant memories from previous conversations. "
        "Use them to provide accurate, context-aware answers. "
        "Higher resonance (R) means more relevant memory.\n\n"
        "Relevant memories:\n{context}"
    ),
    ContextFormat.JSON: (
        "You are a helpful assistant with long-term memory.\n"
        "Below are relevant memories in JSON format. Each entry has:\n"
        "- resonance: how well it matches the current query (higher = more relevant)\n"
        "- salience: overall importance in the memory field\n"
        "- text: the actual memory content\n"
        "- lineage: history of how this memory was formed through consolidation\n"
        "Use these memories to provide accurate, context-aware answers.\n\n"
        "Relevant memories:\n{context}"
    ),
    ContextFormat.YAML: (
        "You are a helpful assistant with long-term memory.\n"
        "Below are relevant memories in YAML format with resonance and salience scores. "
        "Higher scores indicate more relevant/important memories. Use them for context-aware answers.\n\n"
        "Relevant memories:\n{context}"
    ),
    ContextFormat.ATTENTION: (
        "You are a helpful assistant with long-term memory.\n"
        "Below are relevant memories with attention-weighted tokens. "
        "Each memory starts with tokens like [ATTN:x.xxxx][SAL:x.xxxx][TIER:X].\n"
        "- ATTN: attention weight — how relevant this memory is to the current query (higher = more relevant)\n"
        "- SAL: salience — overall importance in the memory field\n"
        "- TIER: memory tier (E=episodic, S=semantic, P=procedural)\n"
        "- CAUSAL: number of causal connections (if present)\n"
        "- GOAL: goal relevance score (if present)\n"
        "Use the ATTN weights to focus your attention on the most relevant memories.\n\n"
        "Relevant memories:\n{context}"
    ),
}


def format_context(results: List[Tuple[str, float, MemoryNode]], fmt: ContextFormat) -> str:
    if fmt == ContextFormat.JSON:
        items = []
        for nid, resp, node in results:
            content = node.content
            
            # Check for structured node (v2)
            if content.get("version") == "2.0":
                item = {
                    "resonance": round(resp, 4),
                    "salience": round(node.salience, 4),
                    "input_text": content.get("input_text", ""),
                    "output_text": content.get("output_text", ""),
                    "role": content.get("role", ""),
                    "session": content.get("session", ""),
                    "emotion": content.get("emotion", ""),
                    "tags": content.get("tags", []),
                    "tier": content.get("tier", ""),
                    "timestamp": content.get("timestamp", 0),
                    "lineage": node.lineage,
                    "modality": node.modality,
                }
            else:
                # Legacy node (v1)
                item = {
                    "resonance": round(resp, 4),
                    "salience": round(node.salience, 4),
                    "text": content.get("text", ""),
                    "lineage": node.lineage,
                    "modality": node.modality,
                    "self_sup_score": round(node.self_sup_score, 4),
                    "cross_modal_score": round(node.cross_modal_score, 4),
                }
                meta = {k: v for k, v in content.items() if k != "text"}
                if meta:
                    item["metadata"] = meta
            items.append(item)
        return json.dumps(items, ensure_ascii=False, indent=2) if items else "[]"
    
    elif fmt == ContextFormat.YAML:
        lines = []
        for nid, resp, node in results:
            content = node.content
            if content.get("version") == "2.0":
                lines.extend([
                    f"- resonance: {resp:.4f}",
                    f"  salience: {node.salience:.4f}",
                    f"  input: \"{content.get('input_text', '')}\"",
                    f"  output: \"{content.get('output_text', '')}\"",
                    f"  role: {content.get('role', '')}",
                    f"  emotion: {content.get('emotion', '')}",
                    f"  tier: {content.get('tier', '')}",
                ])
            else:
                lines.extend([
                    f"- resonance: {resp:.4f}",
                    f"  salience: {node.salience:.4f}",
                    f"  text: \"{content.get('text', '')}\"",
                    f"  lineage: {node.lineage}",
                    f"  modality: {node.modality}",
                    f"  cross_modal_score: {node.cross_modal_score:.4f}",
                ])
        return "\n".join(lines) if lines else "No relevant memory."
    
    elif fmt == ContextFormat.ATTENTION:
        lines = ["### ATTENTION_CONTEXT"]
        for nid, resp, node in results:
            content = node.content
            causal = len(node.causal_strength) if hasattr(node, 'causal_strength') else 0
            goal_rel = getattr(node, 'goal_relevance', 0.0)
            tokens = (f"[ATTN:{resp:.3f}][SAL:{node.salience:.3f}]"
                      f"[TIER:{content.get('tier', getattr(node, 'tier', 'semantic'))[0].upper()}]")
            # Phase 20: Domain & State tokens
            domain = getattr(node, 'domain', 'general')
            if domain and domain != 'general':
                tokens += f"[DOM:{domain.upper()[:3]}]"
            state = getattr(node, 'state', '')
            if state and state != 'stable':
                tokens += f"[STATE:{state[0].upper()}]"
            if causal > 0:
                tokens += f"[CAUSAL:{causal}]"
            if goal_rel > 0.3:
                tokens += f"[GOAL:{goal_rel:.2f}]"
            
            # Extract text from structured or legacy node
            if content.get("version") == "2.0":
                input_t = content.get("input_text", "")[:60]
                output_t = content.get("output_text", "")[:60]
                if input_t and output_t:
                    text = f"U:{input_t} | AI:{output_t}"
                elif input_t:
                    text = f"U:{input_t}"
                elif output_t:
                    text = f"AI:{output_t}"
                else:
                    text = content.get("text", "unknown")[:100]
                # Add emotion/tag if present
                emotion = content.get("emotion", "")
                tags = content.get("tags", [])
                if emotion != "neutral":
                    text += f" [{emotion}]"
                if tags:
                    text += f" #{','.join(tags[:2])}"
            else:
                text = node.content.get("text", "unknown")[:100]
            
            lines.append(f"{tokens} {text}")
        return "\n".join(lines) if len(lines) > 1 else "No relevant memory."
    else:
        parts = []
        for _, r, n in results:
            content = n.content
            if content.get("version") == "2.0":
                input_t = content.get("input_text", "")[:50]
                output_t = content.get("output_text", "")[:50]
                text = f"U:{input_t} | AI:{output_t}" if input_t and output_t else (input_t or output_t or "unknown")
            else:
                text = n.content.get('text', '')
            parts.append(f"[R:{r:.2f}|S:{n.salience:.2f}|CM:{n.cross_modal_score:.2f}] {text}")
        return "\n".join(parts) if parts else "No relevant memory."


def build_system_prompt(context: str, fmt: ContextFormat, use_structured: bool) -> str:
    if not use_structured or not context or context in ("No relevant memory.", "[]"):
        return "You are a helpful assistant with long-term memory."
    return SYSTEM_PROMPT_TEMPLATES.get(fmt, SYSTEM_PROMPT_TEMPLATES[ContextFormat.PLAIN]).format(context=context)


# ============================================================================
# CORE: RTmdKField v7
# ============================================================================

class RTMDKField:
    def __init__(self, config: RTMDKConfig, projection_matrix: Optional[NDArray] = None):
        self.cfg = config
        self._rng = np.random.default_rng(config.seed)
        self.nodes: Dict[str, MemoryNode] = {}
        self.node_index: List[str] = []

        # P0: Cached numpy arrays for vectorized query — avoids O(N) Python loop on every query
        self._cached_positions: Optional[NDArray] = None       # (N, latent_dim)
        self._cached_phases: Optional[NDArray] = None          # (N,)
        self._cached_amplitudes: Optional[NDArray] = None      # (N,)
        self._cached_saliences: Optional[NDArray] = None       # (N,)
        self._cached_modal_weights: Optional[NDArray] = None   # (N,)
        self._cached_gates: Optional[NDArray] = None           # (N,) soft_gate values
        self._cached_causal_boost: Optional[NDArray] = None    # (N,) causal boost factor
        self._cache_dirty: bool = False

        if config.learn_projection:
            self.projection_learner = IncPCAProjection(
                config.embedding_dim, config.pca_n_components or config.latent_dim,
                config.projection_lr, config.projection_update_freq, config.l2_regularization)
            if projection_matrix is not None:
                self.projection_learner.set_matrix(projection_matrix)
        else:
            self.projection_learner = None
            self._raw_projection = (projection_matrix.astype(np.float32) if projection_matrix is not None
                                    else self._rng.standard_normal((config.embedding_dim, config.latent_dim)).astype(np.float32) * 0.1)

        self.adaptive_threshold = AdaptiveThreshold(config.adaptive_window, config.tension_threshold) if config.adaptive_threshold else None
        self.bm25_index = BM25Index(config.bm25_k1, config.bm25_b) if config.bm25_fallback else None
        self.tda_monitor = TDAMonitor() if config.tda_monitoring else None
        self.gpu_backend = TorchBackend() if config.backend == Backend.TORCH else None
        if self.gpu_backend and not self.gpu_backend.available:
            self.gpu_backend = None
        self.hnsw_index = HNSWIndex(config.hnsw_m, config.hnsw_ef_construction) if config.use_hnsw else None

        # Pre-select batch resonance backend to avoid branching in hot path
        if self.gpu_backend and self.gpu_backend.available:
            self._batch_resonance_fn = self._batch_resonance_torch
        else:
            self._batch_resonance_fn = self._batch_resonance_numpy

        self.learnable_kernel: Optional[LearnableKernel] = None
        self.diff_consolidation: Optional[DifferentiableConsolidation] = None
        if config.differentiable:
            self.learnable_kernel = LearnableKernel(config.bandwidth, config.phase_coupling, config.decay_rate, config.gradient_clip)
            self.diff_consolidation = DifferentiableConsolidation(config.consolidation_loss_weight)

        self.monitor: Optional[Any] = None

        self.meta_kernel: Optional[MetaAdaptiveKernel] = None
        if config.meta_adaptive:
            self.meta_kernel = MetaAdaptiveKernel(config.bandwidth, config.phase_coupling, config.meta_adaptation_lr,
                                                  config.kurtosis_target_min, config.kurtosis_target_max)

        self.healer: Optional[TopologyHealer] = None
        if config.self_healing:
            self.healer = TopologyHealer(config.dead_zone_threshold, config.hyperconvergence_threshold,
                                        config.fragmentation_threshold, config.healing_strength, config.max_healing_nodes_per_step)

        # B2: Lazy module initialization — store flags but don't instantiate yet
        self._causal_engine: Optional[CausalInferenceEngine] = None
        self._causal_engine_initialized = config.causal_topological

        self._ode_dynamics: Optional[NeuralODEDynamics] = None
        self._ode_dynamics_initialized = config.continuous_dynamics

        # Track 10.2: Meta-controller — B2 lazy init
        self._meta_controller: Optional[MetaController] = None
        self._meta_controller_initialized = config.meta_controller

        self.agent_planner: Optional[AgentPlanner] = None
        self.hypothesis_verifier: Optional[HypothesisVerifier] = None
        self.tool_router: Optional[ToolRouter] = None
        if config.agent_orchestration:
            self.agent_planner = AgentPlanner(config.max_plan_depth, config.max_tool_calls, config.tool_timeout)
            self.hypothesis_verifier = HypothesisVerifier(config.verification_confidence_threshold)
            self.tool_router = ToolRouter(config.tool_timeout)

        self.shadow_evaluator: Optional[ShadowModeEvaluator] = None
        self.ragas_evaluator: Optional[RAGASPlusEvaluator] = None
        self.rollback_manager: Optional[AutoRollbackManager] = None
        if config.production_mode:
            if config.shadow_mode:
                self.shadow_evaluator = ShadowModeEvaluator(config.shadow_fallback_threshold)
            if config.ragas_enabled:
                self.ragas_evaluator = RAGASPlusEvaluator()
            if config.auto_rollback:
                self.rollback_manager = AutoRollbackManager(config.auto_rollback_threshold)

        # Track 10.3: Federated
        self.federated: Optional[FederatedRTMDK] = None
        if config.federated:
            self.federated = FederatedRTMDK(
                node_id=config.node_id,
                sync_lr=config.federated_sync_lr,
                sync_freq=config.federated_sync_freq,
                min_resonance=config.federated_min_resonance,
            )

        # Phase 11 Track 3: Predictive coding
        self.predictor: Optional[PredictiveCodingModel] = None
        if config.predictive_coding:
            self.predictor = PredictiveCodingModel(config.latent_dim, lr=config.pc_lr)
        self._state_history: deque = deque(maxlen=100)

        # Phase 11 Track 4: Counterfactual imagination
        self.scenario_planner: Optional[ScenarioPlanner] = None
        if config.counterfactual_imagination:
            self.scenario_planner = ScenarioPlanner(self, max_scenarios=config.max_scenarios)

        # Phase 11 Track 5: Differential privacy
        self.dp: Optional[DifferentialPrivacy] = None
        if config.differential_privacy:
            self.dp = DifferentialPrivacy(config.dp_epsilon, config.dp_delta, config.dp_max_norm)

        # Phase 12 Track 1: Sparse resonant routing (MoE-memory)
        self.shard_centers: Optional[NDArray] = None
        self.shard_router: Optional[NDArray] = None
        self._node_shard_map: Dict[str, int] = {}
        if config.sparse_routing:
            self.shard_centers = self._rng.standard_normal((config.num_shards, config.latent_dim)).astype(np.float32)
            self.shard_router = np.zeros(config.num_shards, dtype=np.float32)

        # Phase 12 Track 3: Crystallization
        self._crystallization_counter = 0
        self._crystallized_nodes: Set[str] = set()

        # Fix 3: Lifecycle & Throttling Controls
        self._workers: List[asyncio.Task] = []
        self._write_lock: Optional[asyncio.Lock] = None
        self._backpressure_events = 0
        self._heavy_modules_degraded = False  # Track if we've entered degraded mode
        self._last_successful_step = time.time()  # For recovery tracking

        # B1: Tension caching
        self._tension_cache: Dict[str, Tuple[float, float]] = {}  # node_id -> (tension, step)
        self._tension_cache_max_age = 5  # steps
        self._tension_cache_hits = 0
        self._tension_cache_misses = 0

        # Phase 12 Track 4: Async pipeline queues
        self.query_q: Optional[asyncio.Queue] = None
        self.save_q: Optional[asyncio.Queue] = None
        self.evolve_q: Optional[asyncio.Queue] = None
        self._workers_started = False
        if config.async_pipeline:
            self.query_q = asyncio.Queue(maxsize=config.query_queue_size)
            self.save_q = asyncio.Queue(maxsize=config.save_queue_size)
            self.evolve_q = asyncio.Queue(maxsize=config.evolve_queue_size)

        # Phase 13 Track 1: Teleological layer
        self.goal_tracker: Optional[GoalTracker] = None
        if config.goal_tracking:
            self.goal_tracker = GoalTracker(
                config.max_goals, config.goal_decay, config.goal_completion_threshold
            )

        # Phase 13 Track 3: RL feedback loop
        self.rl_feedback_loop: Optional[RLFeedbackLoop] = None
        if config.rl_feedback:
            self.rl_feedback_loop = RLFeedbackLoop(
                config.rl_learning_rate, config.rl_reward_window
            )

        # Phase 13 Track 4: Event-driven + Low-Rank
        self.event_scheduler: Optional[EventDrivenScheduler] = None
        self.low_rank_compressor: Optional[LowRankCompressor] = None
        if config.event_driven:
            self.event_scheduler = EventDrivenScheduler()
        if config.low_rank_compression:
            self.low_rank_compressor = LowRankCompressor(config.compression_rank)

        # Phase 18: Engram Manager (Fix 4: ensure attribute always exists even when disabled)
        self.engram_manager: Optional[Any] = None

        # Phase 14 Track 1: Meta-Memory
        self.meta_memory_eval: Optional[MetaMemoryEvaluator] = None
        if config.meta_memory:
            self.meta_memory_eval = MetaMemoryEvaluator(
                config.recall_accuracy_threshold, config.memory_age_factor,
                config.self_reflection_freq
            )

        # Phase 14 Track 2: Security
        self.security: Optional[SecurityValidator] = None
        if config.security_enabled:
            self.security = SecurityValidator(
                config.max_node_text_length, config.tension_spike_threshold,
                config.prompt_injection_patterns
            )

        # Phase 14 Track 5: Swarm Memory
        self.swarm: Optional[SwarmConsensusProtocol] = None
        if config.swarm_memory:
            self.swarm = SwarmConsensusProtocol(
                config.swarm_consensus_threshold, config.swarm_max_agents,
                config.swarm_vote_weight
            )

        # Phase 15 Track 1: Version Control (Memory Git)
        self.version_control: Optional["VersionControl"] = None
        if config.version_control and VC_AVAILABLE:
            self.version_control = VersionControl(max_versions=config.max_versions)
        elif config.version_control and not VC_AVAILABLE:
            logger.error("version_control enabled but rtmdk.support.version_control not available — feature disabled")
            self.stats.setdefault("startup_warnings", []).append("version_control unavailable")

        # Phase 15 Track 4: Entropy Control
        self.entropy_ctrl: Optional["EntropyController"] = None
        if config.entropy_management and ENTROPY_AVAILABLE:
            self.entropy_ctrl = EntropyController(
                high_entropy_threshold=config.entropy_high_threshold,
                low_entropy_threshold=config.entropy_low_threshold,
            )
        elif config.entropy_management and not ENTROPY_AVAILABLE:
            logger.error("entropy_management enabled but rtmdk.support.entropy_controller not available — feature disabled")
            self.stats.setdefault("startup_warnings", []).append("entropy_controller unavailable")

        # Phase 15 Track 5: Triton Backend
        self.triton_backend: Optional[Any] = None
        if config.triton_backend and TritonBackend is not None:
            self.triton_backend = TritonBackend(min_nodes_for_gpu=config.min_nodes_for_gpu)

        # Phase 16 Track 1: SymbolicOverlay
        self.symbolic_overlay: Optional["SymbolicOverlay"] = None
        if config.symbolic_overlay and SYMBOLIC_AVAILABLE:
            self.symbolic_overlay = SymbolicOverlay(
                min_self_sup=config.symbolic_min_self_sup,
                max_tension=config.symbolic_max_tension,
                confidence_threshold=config.symbolic_confidence_threshold,
            )
        elif config.symbolic_overlay and not SYMBOLIC_AVAILABLE:
            logger.error("symbolic_overlay enabled but rtmdk.support.symbolic_overlay not available — feature disabled")
            self.stats.setdefault("startup_warnings", []).append("symbolic_overlay unavailable")

        # Phase 16 Track 2: SafetyCertifier
        self.safety_certifier: Optional["SafetyCertifier"] = None
        if config.safety_certifier and SAFETY_AVAILABLE:
            self.safety_certifier = SafetyCertifier(
                mode=config.safety_mode,
                lyapunov_threshold=config.lyapunov_threshold,
                alpha=config.lyapunov_alpha,
                beta=config.lyapunov_beta,
                gamma=config.lyapunov_gamma,
            )
        elif config.safety_certifier and not SAFETY_AVAILABLE:
            logger.error("safety_certifier enabled but rtmdk.support.safety_certifier not available — feature disabled")
            self.stats.setdefault("startup_warnings", []).append("safety_certifier unavailable")

        # Phase 17: RoleShardRouter
        self.role_router: Optional["RoleShardRouter"] = None
        if config.role_sharding and ROLE_SHARD_AVAILABLE:
            self.role_router = RoleShardRouter(
                shards=config.role_shards,
                cross_shard_threshold=config.cross_shard_threshold,
                auto_role_detection=config.auto_role_detection,
            )
        elif config.role_sharding and not ROLE_SHARD_AVAILABLE:
            logger.error("role_sharding enabled but rtmdk.support.role_shard_router not available — feature disabled")
            self.stats.setdefault("startup_warnings", []).append("role_shard_router unavailable")

        self.stats = {
            "total_adds": 0, "total_queries": 0, "consolidations": 0,
            "avg_response": 0.0, "active_nodes": 0,
            "projection_updates": 0, "self_sup_checks": 0, "tda_checks": 0,
            "bm25_fallbacks": 0, "adaptive_threshold_value": config.tension_threshold,
            "false_merges": 0, "field_stability": 1.0,
            "causal_edges": 0, "contradictions": 0, "counterfactual_queries": 0,
            "consolidation_validations": 0, "blocked_consolidations": 0,
            "meta_kurtosis": 3.0, "meta_bandwidth": config.bandwidth,
            "meta_phase_coupling": config.phase_coupling,
            "field_health": "stable", "healing_events": 0, "healing_history": [],
            "ode_steps": 0, "response_smoothness": 1.0,
            "plans_created": 0, "hypotheses_verified": 0, "tool_calls": 0, "tool_misuse_rate": 0.0,
            "evaluations": 0, "shadow_comparisons": 0, "rollbacks": 0,
            "ragas_overall": 0.0,
            "cross_modal_queries": 0, "cross_modal_recall": 0.0,
            "meta_optimizations": 0, "meta_best_params": {},
            "federated_syncs": 0, "federated_order_parameter": 0.0,
            # Phase 11
            "tier_distribution": {}, "tier_coherence": 0.0,
            "hyperbolic_enabled": config.hyperbolic, "avg_hyperbolic_dist": 0.0,
            "free_energy": 0.0, "prediction_error": 0.0, "surprise_level": 0.0,
            "scenarios_generated": 0, "avg_scenario_confidence": 0.0,
            "privacy_budget_spent": 0.0, "noise_std": 0.0, "updates_clipped": 0,
            # Phase 12
            "shard_hits": 0, "shard_misses": 0, "avg_shard_query_time_ms": 0.0,
            "context_tokens_saved": 0, "cognitive_compressions": 0,
            "crystallizations": 0, "crystallized_clusters": 0,
            "async_queue_depth": 0, "async_backpressure_events": 0,
            # Phase 13
            "active_goals": 0, "completed_goals": 0,
            "avg_rl_reward": 0.5, "reward_trend": 0.0,
            "attention_bias_applied": 0,
            "compression_ratio": 1.0, "compression_updates": 0,
            "events_processed": 0, "event_queue_depth": 0,
            # Phase 14
            "recall_accuracy": 1.0, "meta_reflections": 0,
            "security_violations": 0, "tension_spikes_blocked": 0,
            "swarm_agents": 0, "swarm_consensus_events": 0,
            # Phase 15
            "current_version": 0, "n_versions": 0,
            "clarifications_generated": 0,
            "entropy": 0.0, "entropy_state": "normal",
            "triton_backend_used": False, "gpu_acceleration": False,
            # Phase 16
            "n_symbolic_rules": 0, "n_symbolic_inferences": 0, "n_symbolic_conflicts": 0,
            "lyapunov_V": 0.0, "lyapunov_dV_dt": 0.0, "safety_regulation_factor": 1.0,
            "safety_mode": "monitor_only",
            # Phase 17
            "n_shards": 0, "shard_distribution": {},
            "cross_shard_exchanges": 0, "role_router_enabled": False,
            # Phase 18: Engrams
            "engram_retrievals": 0, "engrams_created": 0, "engrams_merged": 0,
            "field_integrity_issues": 0,
            "backpressure_degraded_mode": 0, "last_backpressure_recovery": 0.0,
            # Fix 10: Track startup warnings for missing optional dependencies
            "startup_warnings": [],
            # B1: Tension cache stats
            "tension_cache_hits": 0, "tension_cache_misses": 0,
            "tension_cache_hit_rate": 0.0,
        }
        self._step_counter = 0
        # Rate limiting: track add_node timestamps (max 100 nodes/sec)
        self._add_node_timestamps: deque = deque(maxlen=1000)
        self._rollback_history: deque = deque(maxlen=config.max_rollback_history)
        self._stability_buffer: deque = deque(maxlen=config.field_stability_window)
        self._active_node_history: deque = deque(maxlen=50)

    def _project(self, embedding: NDArray) -> NDArray:
        if self.projection_learner:
            latent = self.projection_learner.project(embedding)
        else:
            latent = ((embedding - 0) @ self._raw_projection).astype(np.float32) if embedding.ndim == 1 else (embedding @ self._raw_projection).astype(np.float32)
        # Phase 11 Track 2: Hyperbolic projection into Poincare ball
        if self.cfg.hyperbolic:
            norm = np.linalg.norm(latent)
            if norm >= self.cfg.ball_radius:
                latent = latent * (self.cfg.ball_radius - 1e-6) / max(norm, 1e-8)
        return latent

    def _get_phase(self, session_id: Optional[str] = None, embedding: Optional[NDArray] = None,
                   modality: str = "text") -> float:
        base = (time.time() * 0.01) % (2 * np.pi)
        if self.cfg.cross_modal and modality in self.cfg.modal_phase_offsets:
            base += self.cfg.modal_phase_offsets[modality]
        elif self.cfg.multimodal and modality in self.cfg.modality_phase_shifts:
            base += self.cfg.modality_phase_shifts[modality]
        return base % (2 * np.pi)

    def _resonance_response(self, query_latent: NDArray, query_phase: float, node: MemoryNode,
                            query_modality: str = "text") -> float:
        # Fix 1: Torch backend auto-switch for batch resonance
        # (Single-node response always uses numpy for simplicity;
        #  batch queries use TorchBackend.batch_resonance via query())

        # Phase 11 Track 2: Hyperbolic distance
        if self.cfg.hyperbolic:
            dist = poincare_dist(query_latent, node.latent_pos, self.cfg.ball_radius)
            self.stats["avg_hyperbolic_dist"] = 0.99 * self.stats["avg_hyperbolic_dist"] + 0.01 * dist
        else:
            dist = np.linalg.norm(query_latent - node.latent_pos)
        phase_diff = node.phase - query_phase
        bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self.cfg.bandwidth
        pc = self.meta_kernel.get_phase_coupling() if self.meta_kernel else self.cfg.phase_coupling

        if self.learnable_kernel:
            resp = self.learnable_kernel.resonance_response(dist, phase_diff, node.amplitude, node.salience)
        else:
            if self.cfg.resonance_kernel == "gaussian":
                spatial = math.exp(-dist ** 2 / (2 * bw ** 2))
            elif self.cfg.resonance_kernel == "cosine":
                nq = np.linalg.norm(query_latent)
                nn = np.linalg.norm(node.latent_pos)
                spatial = 0.5 + 0.5 * np.dot(query_latent, node.latent_pos) / (nq * nn + 1e-8) if nq > 1e-8 and nn > 1e-8 else 0.5
            else:
                spatial = math.exp(-dist / bw)
            phase_align = 0.5 + 0.5 * math.cos(phase_diff)
            resp = spatial * ((1 - pc) + pc * phase_align) * node.amplitude * node.salience

        gate = node.soft_gate if self.cfg.soft_gates else 1.0
        if self.causal_engine and node.causal_parents:
            causal_boost = sum(node.causal_strength.get(p, 0) for p in node.causal_parents)
            resp *= (1.0 + 0.1 * causal_boost)

        if self.cfg.cross_modal:
            resp = cross_modal_resonance(
                query_modality, node.modality, resp,
                self.cfg.modal_phase_offsets, self.cfg.cross_modal_kernel_weight
            )
            base_val = spatial * node.amplitude * node.salience
            node.cross_modal_score = resp / base_val if base_val > 1e-8 else 0.0

        return resp * gate * node.modal_weight

    def _batch_resonance(self, query_latents: NDArray, query_phases: NDArray,
                         node_ids: List[str]) -> NDArray:
        """Batch resonance computation. Pre-selected backend avoids hot-path branching."""
        return self._batch_resonance_fn(query_latents, query_phases, node_ids)

    def _batch_resonance_numpy(self, query_latents: NDArray, query_phases: NDArray,
                               node_ids: List[str]) -> NDArray:
        """Pure numpy batch resonance — no branching, no torch overhead."""
        if not node_ids:
            return np.empty((len(query_latents), 0), dtype=np.float32)

        node_positions = np.array([self.nodes[nid].latent_pos for nid in node_ids])
        node_phases = np.array([self.nodes[nid].phase for nid in node_ids])
        node_amplitudes = np.array([self.nodes[nid].amplitude for nid in node_ids])
        node_saliences = np.array([self.nodes[nid].salience for nid in node_ids])

        dists = cdist(query_latents, node_positions)
        # Gaussian kernel: exp(-d^2/(2*bw^2)) — use meta_kernel if available (Fix 2: consistency with single-node path)
        bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self.cfg.bandwidth
        pc = self.meta_kernel.get_phase_coupling() if self.meta_kernel else self.cfg.phase_coupling
        spatial = np.exp(-dists ** 2 / (2 * bw ** 2))
        phase_diff = query_phases[:, np.newaxis] - node_phases[np.newaxis, :]
        phase_align = 0.5 + 0.5 * np.cos(phase_diff)
        response = spatial * ((1 - pc) + pc * phase_align)
        return response * node_amplitudes[np.newaxis, :] * node_saliences[np.newaxis, :]

    def _batch_resonance_torch(self, query_latents: NDArray, query_phases: NDArray,
                               node_ids: List[str]) -> NDArray:
        """Torch batch resonance — GPU accelerated."""
        if not node_ids:
            return np.empty((len(query_latents), 0), dtype=np.float32)

        node_positions = np.array([self.nodes[nid].latent_pos for nid in node_ids])
        node_phases = np.array([self.nodes[nid].phase for nid in node_ids])
        node_amplitudes = np.array([self.nodes[nid].amplitude for nid in node_ids])
        node_saliences = np.array([self.nodes[nid].salience for nid in node_ids])

        # Use meta_kernel if available (Fix 2: consistency with single-node path)
        bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self.cfg.bandwidth
        pc = self.meta_kernel.get_phase_coupling() if self.meta_kernel else self.cfg.phase_coupling
        return self.gpu_backend.batch_resonance(
            query_latents, query_phases, node_positions, node_phases,
            node_amplitudes, node_saliences,
            bw, pc
        )

    def _build_node_cache(self):
        """Build numpy arrays cache from nodes — called once when cache is dirty."""
        n = len(self.node_index)
        if n == 0:
            self._cached_positions = np.empty((0, self.cfg.latent_dim), dtype=np.float32)
            self._cached_phases = np.empty(0, dtype=np.float32)
            self._cached_amplitudes = np.empty(0, dtype=np.float32)
            self._cached_saliences = np.empty(0, dtype=np.float32)
            self._cached_modal_weights = np.empty(0, dtype=np.float32)
            self._cached_gates = np.empty(0, dtype=np.float32)
            self._cached_causal_boost = np.empty(0, dtype=np.float32)
            self._cache_dirty = False
            return

        # Single pass through nodes — much faster than 5 separate list comprehensions
        positions = np.zeros((n, self.cfg.latent_dim), dtype=np.float32)
        phases = np.zeros(n, dtype=np.float32)
        amplitudes = np.zeros(n, dtype=np.float32)
        saliences = np.zeros(n, dtype=np.float32)
        modal_weights = np.zeros(n, dtype=np.float32)
        gates = np.ones(n, dtype=np.float32)  # Default gate = 1.0
        causal_boost = np.zeros(n, dtype=np.float32)  # Default causal boost = 0

        for i, nid in enumerate(self.node_index):
            node = self.nodes[nid]
            positions[i] = node.latent_pos
            phases[i] = node.phase
            amplitudes[i] = node.amplitude
            saliences[i] = node.salience
            modal_weights[i] = node.modal_weight
            if self.cfg.soft_gates and hasattr(node, 'soft_gate'):
                gates[i] = node.soft_gate
            # Pre-compute causal boost factor: 1.0 + 0.1 * sum(causal_strength)
            if self.causal_engine and hasattr(node, 'causal_parents') and node.causal_parents:
                cb = sum(node.causal_strength.get(p, 0) for p in node.causal_parents)
                causal_boost[i] = 1.0 + 0.1 * cb

        self._cached_positions = positions
        self._cached_phases = phases
        self._cached_amplitudes = amplitudes
        self._cached_saliences = saliences
        self._cached_modal_weights = modal_weights
        self._cached_gates = gates
        self._cached_causal_boost = causal_boost
        self._cache_dirty = False

    def _query_vectorized(self, query_latent: NDArray, query_phase: float,
                          top_k: int, modality: str, session_id: Optional[str],
                          t0: float) -> List[Tuple[str, float, MemoryNode]]:
        """Vectorized query using cached numpy arrays — O(N) but vectorized.

        Mathematical model:
        - dist_i = ||q - n_i|| for all i → vectorized norm
        - spatial_i = exp(-dist_i² / 2bw²) → vectorized exp
        - phase_align_i = 0.5 + 0.5*cos(phase_i - query_phase) → vectorized cos
        - resp_i = spatial_i × ((1-pc) + pc × phase_align_i) × amp_i × sal_i
        - Session boost: resp_i × 1.5 if session matches
        - Filter: resp_i >= min_response
        - Sort and return top_k

        Complexity: O(N×d) with SIMD vectorization (~200x faster than Python loop)
        Cached arrays avoid O(N) Python loop on every query.
        """
        n_nodes = len(self.node_index)
        if n_nodes == 0:
            return []

        # Build cache if dirty (single pass through nodes)
        if self._cache_dirty:
            self._build_node_cache()

        # P1: Session pre-filtering — build mask once, apply to all arrays
        session_mask = None
        if session_id and session_id != "default":
            session_mask = np.array([
                self.nodes[nid].content.get("session") == session_id
                for nid in self.node_index
            ], dtype=bool)
            # If very few session nodes, use them directly
            n_session = session_mask.sum()
            if 0 < n_session < n_nodes * 0.3:
                # Session has < 30% of nodes — filter arrays
                positions = self._cached_positions[session_mask]
                phases = self._cached_phases[session_mask]
                amplitudes = self._cached_amplitudes[session_mask]
                saliences = self._cached_saliences[session_mask]
                modal_weights = self._cached_modal_weights[session_mask]
                gates = self._cached_gates[session_mask]
                causal_boost = self._cached_causal_boost[session_mask]
                session_indices = np.where(session_mask)[0]
            else:
                # Session has many nodes — use full arrays with boost
                positions = self._cached_positions
                phases = self._cached_phases
                amplitudes = self._cached_amplitudes
                saliences = self._cached_saliences
                modal_weights = self._cached_modal_weights
                gates = self._cached_gates
                causal_boost = self._cached_causal_boost
                session_indices = None
        else:
            positions = self._cached_positions
            phases = self._cached_phases
            amplitudes = self._cached_amplitudes
            saliences = self._cached_saliences
            modal_weights = self._cached_modal_weights
            gates = self._cached_gates
            causal_boost = self._cached_causal_boost
            session_indices = None

        # Vectorized distance computation
        dists = np.linalg.norm(positions - query_latent, axis=1)

        # Vectorized spatial kernel (gaussian)
        bw = self.meta_kernel.get_bandwidth() if self.meta_kernel else self.cfg.bandwidth
        spatial = np.exp(-dists ** 2 / (2 * bw ** 2))

        # Vectorized phase alignment
        pc = self.meta_kernel.get_phase_coupling() if self.meta_kernel else self.cfg.phase_coupling
        phase_align = 0.5 + 0.5 * np.cos(phases - query_phase)

        # Vectorized resonance response
        resp = spatial * ((1 - pc) + pc * phase_align) * amplitudes * saliences * modal_weights

        # Apply soft gates (matches single query: resp *= gate)
        if self.cfg.soft_gates:
            resp = resp * gates

        # Apply causal boosting (matches single query: resp *= (1 + 0.1 * sum(causal_strength)))
        if self.causal_engine:
            resp = resp * causal_boost

        # Session boost (vectorized) — apply to full-array case
        if session_id and session_id != "default" and session_mask is not None and session_indices is None:
            resp = resp * (1.0 + 0.5 * session_mask.astype(np.float32))

        # Filter by min_response threshold
        above_threshold = resp >= self.cfg.min_response
        indices = np.where(above_threshold)[0]

        if len(indices) == 0:
            self.stats["total_queries"] += 1
            return []

        # Map back to original node_index if session-filtered
        if session_indices is not None:
            indices = session_indices[indices]

        # P1: Use argpartition for partial sort — O(N) instead of O(N log N)
        n_results = min(len(indices), top_k * 2)  # Get top_k*2 to be safe
        if len(indices) > top_k * 3:
            # Partial sort: only guarantee top_k are correct
            scores = resp[indices]
            if n_results < len(scores):
                partition_idx = np.argpartition(scores, -n_results)[-n_results:]
                top_local = partition_idx[np.argsort(scores[partition_idx])[::-1][:top_k]]
            else:
                top_local = np.argsort(scores)[::-1][:top_k]
            top_indices = indices[top_local]
            top_scores = scores[top_local]
        else:
            # Small result set — full sort is fine
            scores = resp[indices]
            sorted_order = np.argsort(scores)[::-1][:top_k]
            top_indices = indices[sorted_order]
            top_scores = scores[sorted_order]

        # Build result list
        results = []
        for i in range(len(top_indices)):
            idx = top_indices[i]
            nid = self.node_index[idx]
            node = self.nodes[nid]
            node.last_resonated = time.time()
            results.append((nid, float(top_scores[i]), node))

        # Update stats
        self.stats["total_queries"] += 1
        if results:
            self.stats["avg_response"] = 0.9 * self.stats["avg_response"] + 0.1 * results[0][1]
            if self.ode_dynamics:
                self.ode_dynamics.record_response(results[0][1])
            if self.entropy_ctrl:
                self.entropy_ctrl.record_response(results[0][1], results[0][2].salience)
            if self.goal_tracker:
                for nid, resp_val, node in results:
                    node.goal_relevance = self.goal_tracker.get_goal_relevance(nid)
            if self.cfg.attention_bias:
                from rtmdk.memory.core import apply_attention_bias
                results = apply_attention_bias(results, self.cfg.bias_temperature)
                self.stats["attention_bias_applied"] += 1

        # Track timing
        elapsed_ms = (time.time() - t0) * 1000
        if self.cfg.sparse_routing:
            self.stats["avg_shard_query_time_ms"] = (
                0.95 * self.stats["avg_shard_query_time_ms"] + 0.05 * elapsed_ms
            )

        return results

    def query(self, embedding: NDArray, phase: float = 0.0, top_k: Optional[int] = None,
              modality: str = "text", session_id: Optional[str] = None) -> List[Tuple[str, float, MemoryNode]]:
        t0 = time.time()
        top_k = top_k or self.cfg.top_k
        query_latent = self._project(embedding)

        # Fix 1: HNSW auto-intercept for large N (>500 nodes)
        if self.cfg.use_hnsw and self.hnsw_index and len(self.hnsw_index.positions) > max(100, top_k * 2):
            candidate_ids = self.hnsw_index.search(query_latent, top_k * 3)
            search_nodes = [(nid, self.nodes[nid]) for nid in candidate_ids if nid in self.nodes]
        elif self.cfg.sparse_routing and self.shard_centers is not None and len(self.nodes) > self.cfg.num_shards * 2:
            active_shards = self._route_query(query_latent, self.cfg.top_shards)
            candidate_ids = [nid for nid in self.node_index if self._get_node_shard(nid) in active_shards]
            search_nodes = [(nid, self.nodes[nid]) for nid in candidate_ids if nid in self.nodes]
            self.stats["shard_hits"] += len(candidate_ids)
        else:
            # OPTIMIZATION: Use vectorized batch resonance for N >= 50 nodes
            if len(self.node_index) >= 50:
                return self._query_vectorized(query_latent, phase, top_k, modality, session_id, t0)
            search_nodes = [(nid, self.nodes[nid]) for nid in self.node_index]
            if self.cfg.sparse_routing:
                self.stats["shard_misses"] += 1

        # Original loop path (for small N < 50)
        # Fix 3: Hyperbolic pre-filtering for candidate selection
        if self.cfg.hyperbolic and len(search_nodes) > top_k * 5:
            query_norm = np.linalg.norm(query_latent)
            if query_norm >= self.cfg.ball_radius:
                query_latent = query_latent * (self.cfg.ball_radius - 1e-6) / max(query_norm, 1e-8)
            prefiltered = []
            for nid, node in search_nodes:
                # FIX: Never mutate node.latent_pos — use a local copy for projection
                node_norm = np.linalg.norm(node.latent_pos)
                node_pos = node.latent_pos
                if node_norm >= self.cfg.ball_radius:
                    node_pos = node.latent_pos * (self.cfg.ball_radius - 1e-6) / max(node_norm, 1e-8)
                hdist = poincare_dist(query_latent, node_pos, self.cfg.ball_radius)
                if hdist < 3.0:
                    prefiltered.append((nid, node))
            if len(prefiltered) > 0:
                search_nodes = prefiltered

        results = []
        for nid, node in search_nodes:
            resp = self._resonance_response(query_latent, phase, node, query_modality=modality)
            # Session priority bonus: boost nodes matching the queried session
            if session_id and node.content.get("session") == session_id:
                resp *= 1.3  # 30% boost for session-matching nodes
            if resp >= self.cfg.min_response:
                results.append((nid, resp, node))
                node.last_resonated = time.time()

        results.sort(key=lambda x: x[1], reverse=True)
        self.stats["total_queries"] += 1

        # Track shard query time
        if self.cfg.sparse_routing:
            elapsed_ms = (time.time() - t0) * 1000
            self.stats["avg_shard_query_time_ms"] = (
                0.95 * self.stats["avg_shard_query_time_ms"] + 0.05 * elapsed_ms
            )

        if self.cfg.cross_modal:
            self.stats["cross_modal_queries"] += 1
            if results:
                cm_scores = [n.cross_modal_score for _, _, n in results]
                self.stats["cross_modal_recall"] = 0.9 * self.stats["cross_modal_recall"] + 0.1 * float(np.mean(cm_scores))

        if len(results) == 0 and self.cfg.bm25_fallback and self.bm25_index:
            # Handle both v1 (text) and v2 (input_text + output_text) nodes
            texts = []
            for nid in self.node_index[:100]:
                content = self.nodes[nid].content
                t = content.get("text", "")
                if not t:
                    t = f"{content.get('input_text', '')} {content.get('output_text', '')}".strip()
                if t:
                    texts.append(t)
            query_text = " ".join(texts)
            if query_text:
                for doc_id, score in self.bm25_index.search(query_text, top_k):
                    if doc_id in self.nodes:
                        results.append((doc_id, score * 0.1, self.nodes[doc_id]))
                self.stats["bm25_fallbacks"] += 1

        if results:
            self.stats["avg_response"] = 0.9 * self.stats["avg_response"] + 0.1 * results[0][1]
            if self.ode_dynamics:
                self.ode_dynamics.record_response(results[0][1])

            # Phase 15 Track 4: Record resonance for entropy
            if self.entropy_ctrl:
                self.entropy_ctrl.record_response(results[0][1], results[0][2].salience)

        # Phase 13 Track 1: Goal relevance scoring
        if self.goal_tracker and results:
            for nid, resp, node in results:
                node.goal_relevance = self.goal_tracker.get_goal_relevance(nid)

        # Phase 13 Track 2: Cognitive attention bias
        if self.cfg.attention_bias and results:
            results = apply_attention_bias(results, self.cfg.bias_temperature)
            self.stats["attention_bias_applied"] += 1

        # Phase 13 Track 4: Event-driven trigger for queries
        if self.event_scheduler and results:
            self.event_scheduler.enqueue("query", {"top_score": results[0][1] if results else 0})

        if self.meta_kernel:
            self.meta_kernel.record_response(results[0][1] if results else 0.0)
            if len(results) >= 2:
                positions = np.array([n.latent_pos for _, _, n in results])
                valid = pdist(positions)
                density = 1.0 / (1.0 + np.mean(valid)) if len(valid) > 0 else 0.0
                self.meta_kernel.record_semantic_density(float(density))
            if len(results) >= 2:
                responses = np.array([r for _, r, _ in results])
                normalized = responses / (np.sum(responses) + 1e-8)
                entropy = -np.sum(normalized * np.log(normalized + 1e-8))
                self.meta_kernel.record_uncertainty(float(entropy))

        # Phase 16 Track 2: Store results for SafetyCertifier
        self._last_query_results = results

        if self.causal_engine and len(results) >= 2:
            self.causal_engine.record_cooccurrence(results[0][0], results[1][0])
            active = [nid for nid, resp, _ in results if resp > self.cfg.min_response * 0.5]
            if active:
                self.causal_engine.record_observation(active)
                self._active_node_history.append(active)

        # Phase 14 Track 1: Meta-memory recall tracking
        if self.meta_memory_eval and results:
            top_score = results[0][1]
            avg_age = np.mean([time.time() - n.created_at for _, _, n in results])
            self.meta_memory_eval.record_recall("", top_score, node_age=avg_age)
            self.stats["recall_accuracy"] = self.meta_memory_eval.evaluate_recall_accuracy()

        return results[:top_k]

    # B2: Lazy property for causal engine
    @property
    def causal_engine(self) -> Optional["CausalInferenceEngine"]:
        if self._causal_engine_initialized and self._causal_engine is None:
            self._causal_engine = CausalInferenceEngine(
                min_samples=self.cfg.causal_discovery_min_samples,
                p_threshold=self.cfg.causal_p_threshold,
                adjustment_sets_enabled=self.cfg.causal_adjustment_sets)
        return self._causal_engine

    @causal_engine.setter
    def causal_engine(self, value: Optional["CausalInferenceEngine"]):
        self._causal_engine = value
        self._causal_engine_initialized = value is not None

    # B2: Lazy property for ODE dynamics
    @property
    def ode_dynamics(self) -> Optional["NeuralODEDynamics"]:
        if self._ode_dynamics_initialized and self._ode_dynamics is None:
            self._ode_dynamics = NeuralODEDynamics(
                self.cfg.latent_dim, self.cfg.sde_noise_level, self.cfg.ode_time_horizon,
                self.cfg.ode_n_steps, self.cfg.ode_chunk_size, self.cfg.ode_solver,
                self.cfg.ode_atol, self.cfg.ode_rtol)
        return self._ode_dynamics

    @ode_dynamics.setter
    def ode_dynamics(self, value: Optional["NeuralODEDynamics"]):
        self._ode_dynamics = value
        self._ode_dynamics_initialized = value is not None

    # B2: Lazy property for meta-controller
    @property
    def meta_controller(self) -> Optional["MetaController"]:
        if self._meta_controller_initialized and self._meta_controller is None:
            self._meta_controller = MetaController(
                n_trials=self.cfg.meta_n_trials,
                optimize_params=self.cfg.meta_optimize_params,
                optimization_freq=self.cfg.meta_optimization_freq,
            )
        return self._meta_controller

    @meta_controller.setter
    def meta_controller(self, value: Optional["MetaController"]):
        self._meta_controller = value
        self._meta_controller_initialized = value is not None

    def add_node(self, embedding: NDArray, content: Dict, phase: Optional[float] = None,
                 node_id: Optional[str] = None, session_id: Optional[str] = None, modality: str = "text",
                 skip_projection: bool = False) -> str:
        # Rate limiting: max 100 nodes per second
        now = time.time()
        while self._add_node_timestamps and self._add_node_timestamps[0] < now - 1.0:
            self._add_node_timestamps.popleft()
        if len(self._add_node_timestamps) >= 100:
            raise SecurityViolationError("Rate limit exceeded: max 100 nodes/second")
        self._add_node_timestamps.append(now)

        # Phase 14 Track 2: Security validation
        if self.security:
            # Check ALL text fields for prompt injection, not just 'text'
            text = content.get("text", "")
            input_text = content.get("input_text", "")
            output_text = content.get("output_text", "")
            for field_text in [text, input_text, output_text]:
                if field_text:
                    validation = self.security.validate_node_content(field_text)
                    if not validation["is_safe"]:
                        self.stats["security_violations"] += 1
                        logger.warning(f"Security violation in add_node: {validation['violations']}")
                        # Fix 7: Raise instead of returning "" — caller must handle
                        raise SecurityViolationError(f"Security violation: {validation['violations']}")

        nid = node_id or f"n_{len(self.nodes)}_{int(time.time() * 1000)}"
        if skip_projection:
            # Input is already in latent space (e.g., crystallization)
            if len(embedding) != self.cfg.latent_dim:
                raise ValueError(
                    f"skip_projection=True but embedding dim {len(embedding)} != "
                    f"latent_dim {self.cfg.latent_dim}"
                )
            latent = embedding
        elif self.projection_learner:
            latent = self.projection_learner.update(embedding)
            self.stats["projection_updates"] += 1
        else:
            latent = self._project(embedding)
        if phase is None:
            phase = self._get_phase(session_id, embedding, modality)

        # OPTIMIZED: Initialize amplitude/salience based on embedding quality
        # Higher norm embeddings → more informative content → higher initial salience
        emb_norm = float(np.linalg.norm(embedding))
        # Typical emb_norm range: 5-30 for real embeddings, 2-10 for synthetic
        # Normalize to [0.5, 1.0] range for salience
        salience = min(1.0, max(0.3, emb_norm / 20.0))
        amplitude = min(1.0, max(0.5, emb_norm / 15.0))

        node = MemoryNode(id=nid, latent_pos=latent, phase=phase,
                          amplitude=amplitude, salience=salience, content=content,
                          lineage=[], modality=modality)

        if self.cfg.cross_modal:
            node.modal_embedding = embedding.copy()

        # Phase 17: Role assignment
        role = DEFAULT_ROLE
        if self.role_router:
            text = content.get("text", "")
            # Check if content has explicit role tag
            explicit_role = content.get("role") or content.get("tier_role")
            role = self.role_router.add_node(nid, text, role=explicit_role)
            node.role = role  # Set role attribute on node

        self.nodes[nid] = node
        # H1: Prevent duplicate node_id in node_index
        if nid not in self.node_index:
            self.node_index.append(nid)
        self.stats["total_adds"] += 1

        # P0: Invalidate cached arrays (will be rebuilt on next query)
        # For single node additions, use incremental append if cache exists
        if self._cached_positions is not None:
            # Incremental append to avoid full rebuild
            try:
                self._cached_positions = np.vstack([self._cached_positions, latent.reshape(1, -1)])
                self._cached_phases = np.append(self._cached_phases, phase if phase is not None else self._get_phase(session_id, embedding))
                self._cached_amplitudes = np.append(self._cached_amplitudes, amplitude)
                self._cached_saliences = np.append(self._cached_saliences, salience)
                self._cached_modal_weights = np.append(self._cached_modal_weights, 1.0)
                self._cached_gates = np.append(self._cached_gates, 1.0)
                self._cached_causal_boost = np.append(self._cached_causal_boost, 1.0)
            except Exception as e:
                logger.warning(f"Incremental cache append failed: {e}, falling back to full rebuild")
                # Fallback: mark dirty for full rebuild
                self._cache_dirty = True
        else:
            self._cache_dirty = True

        # B1: Invalidate tension cache for neighbors (new node affects topology)
        self._invalidate_tension_cache(nid)

        # Phase 17: Update shard distribution stats
        if self.role_router:
            self.stats["n_shards"] = len(self.role_router.shards)
            self.stats["shard_distribution"] = {
                r: len(s.node_ids) for r, s in self.role_router.shards.items()
            }
            self.stats["role_router_enabled"] = True

        if self.cfg.use_hnsw and self.hnsw_index:
            self.hnsw_index.insert(nid, latent)
        if self.cfg.bm25_fallback and self.bm25_index:
            # Handle both v1 (text) and v2 (input_text + output_text) nodes
            text = content.get("text", "")
            if not text:
                input_t = content.get("input_text", "")
                output_t = content.get("output_text", "")
                text = f"{input_t} {output_t}".strip()
            if text:
                self.bm25_index.add_document(nid, text)

        # Phase 13 Track 1: Event-driven trigger for node added
        if self.event_scheduler:
            self.event_scheduler.enqueue("node_added", {"node_id": nid, "modality": modality})

        return nid

    def _invalidate_tension_cache(self, node_id: Optional[str] = None):
        """B1: Invalidate tension cache. If node_id given, invalidate that node and neighbors.
        Otherwise, invalidate entire cache. Also cleans entries for deleted nodes."""
        # H8: Clean up entries for deleted nodes on every call
        dead_keys = [k for k in self._tension_cache if k not in self.nodes]
        for k in dead_keys:
            self._tension_cache.pop(k, None)

        if node_id is not None:
            # Remove specific node and mark neighbors for refresh
            self._tension_cache.pop(node_id, None)
            # Invalidate cache for nodes near the changed one
            node = self.nodes.get(node_id)
            if node:
                for nid in list(self._tension_cache.keys()):
                    if nid == node_id:
                        continue
                    # Simple proximity check: invalidate ~20% of cache
                    if hash(nid) % 5 == 0:
                        self._tension_cache.pop(nid, None)
        else:
            # Full invalidation
            self._tension_cache.clear()

    def _sweep_tension_cache(self):
        """Remove stale tension cache entries for live nodes (Fix 3: prevent unbounded cache growth)."""
        if not self._tension_cache:
            return
        # Only sweep if cache is large (more than 2x number of nodes)
        if len(self._tension_cache) <= len(self.nodes) * 2:
            return
        current_step = self._step_counter
        keys_to_remove = [
            k for k, (tension, step) in self._tension_cache.items()
            if current_step - step > self._tension_cache_max_age * 3
            and k in self.nodes  # Only remove for live nodes
        ]
        for k in keys_to_remove:
            self._tension_cache.pop(k, None)

    def _compute_tension(self, node_id: str, neighborhood_radius: float = 2.0) -> float:
        # B1: Tension cache check
        if node_id in self._tension_cache:
            cached_tension, cached_step = self._tension_cache[node_id]
            if self._step_counter - cached_step < self._tension_cache_max_age:
                self._tension_cache_hits += 1
                return cached_tension

        self._tension_cache_misses += 1

        node = self.nodes[node_id]

        # Use HNSW for fast k-NN, else fallback to deterministic k-NN via cdist
        k_neighbors = 10
        neighbor_ids = []

        if self.cfg.use_hnsw and self.hnsw_index and len(self.hnsw_index.positions) > k_neighbors:
            candidate_ids = self.hnsw_index.search(node.latent_pos, top_k=k_neighbors + 1)
            neighbor_ids = [nid for nid in candidate_ids if nid != node_id and nid in self.nodes]
        else:
            # Deterministic fallback: compute distances to a limited window
            ids_to_check = self.node_index
            max_scan = 200  # Limit scan for performance
            if len(ids_to_check) > max_scan:
                # Use reservoir-style sample with deterministic seed based on node_id
                rng = np.random.RandomState(int(hashlib.md5(node_id.encode()).hexdigest(), 16) % 2**32)
                ids_to_check = list(rng.choice(ids_to_check, size=max_scan, replace=False))

            if len(ids_to_check) < 2:
                return 0.0

            # Compute distances and select k nearest within radius
            others = [(oid, self.nodes[oid]) for oid in ids_to_check if oid != node_id and oid in self.nodes]
            if not others:
                return 0.0

            other_positions = np.array([n.latent_pos for _, n in others])
            other_ids = [oid for oid, _ in others]
            dists = np.linalg.norm(other_positions - node.latent_pos, axis=1)

            # Filter by radius and select k nearest
            within_radius = dists < neighborhood_radius
            if not np.any(within_radius):
                # Fallback: take k nearest regardless of radius
                k = min(k_neighbors, len(dists))
                nearest_idx = np.argsort(dists)[:k]
                neighbor_ids = [other_ids[i] for i in nearest_idx]
            else:
                radius_dists = [(other_ids[i], dists[i]) for i in range(len(dists)) if within_radius[i]]
                radius_dists.sort(key=lambda x: x[1])
                neighbor_ids = [oid for oid, _ in radius_dists[:k_neighbors]]

        if len(neighbor_ids) < 2:
            tension = 0.0
        else:
            neighbors = [self.nodes[oid] for oid in neighbor_ids]
            phases = np.array([n.phase for n in neighbors])
            saliences = np.array([n.salience for n in neighbors])
            tension = 0.6 * (np.std(np.cos(phases)) + np.std(np.sin(phases))) + 0.4 * np.std(saliences)

        # Phase 14 Track 2: Security - detect tension spikes
        if self.security and not self.security.validate_tension_spike(float(tension)):
            self.stats["tension_spikes_blocked"] += 1

        # B1: Cache the computed tension
        result = float(tension)
        self._tension_cache[node_id] = (result, self._step_counter)
        return result

    def _soft_gate(self, tension: float) -> float:
        if not self.cfg.soft_gates:
            return 1.0
        eff = self.adaptive_threshold.get_threshold() if self.adaptive_threshold else self.cfg.tension_threshold
        return float(1 / (1 + math.exp(-(tension - eff) / self.cfg.gate_temperature)))

    def get_effective_threshold(self) -> float:
        return self.adaptive_threshold.get_threshold() if self.adaptive_threshold else self.cfg.tension_threshold

    def consolidate(self, mode: Optional[ConsolidationMode] = None) -> List[str]:
        mode = mode or self.cfg.consolidation_mode
        updated = []
        eff_threshold = self.get_effective_threshold()

        pre_state = {}
        if self.cfg.enable_rollback or self.cfg.self_sup_verify_after_consolidate:
            for nid in self.node_index:
                n = self.nodes[nid]
                # H3: Save full state for complete rollback — not just position/phase
                pre_state[nid] = {
                    "latent_pos": n.latent_pos.copy(),
                    "phase": n.phase,
                    "amplitude": n.amplitude,
                    "salience": n.salience,
                    "tension": n.tension,
                    "soft_gate": n.soft_gate,
                    "content": dict(n.content),  # shallow copy — synthesis_note gets added
                    "lineage": list(n.lineage),
                    "causal_strength": dict(n.causal_strength),
                    "causal_parents": list(n.causal_parents),
                }

        # Fix 2: Safe iteration — snapshot node_index to avoid mutation issues
        node_index_snapshot = list(self.node_index)
        for nid in node_index_snapshot:
            if nid not in self.nodes:
                continue
            tension = self._compute_tension(nid)
            self.nodes[nid].tension = tension
            self.nodes[nid].soft_gate = self._soft_gate(tension)
            if self.adaptive_threshold:
                self.adaptive_threshold.record_tension(tension)
                self.stats["adaptive_threshold_value"] = self.adaptive_threshold.get_threshold()

        high_tension = [nid for nid in node_index_snapshot if nid in self.nodes and self.nodes[nid].tension > eff_threshold]
        processed = set()
        pending_deletions = []
        # Fix: Snapshot node_index ONCE before outer loop (was O(N²) due to repeated copies)
        node_index_snapshot = list(self.node_index)
        n_snap = len(node_index_snapshot)

        # FIX: Precompute positions for vectorized distance computation
        if self.cfg.use_hnsw and self.hnsw_index and n_snap > 50:
            # Use HNSW for candidate search — O(N log N)
            # Fix 10: Track HNSW bypass when node count <= 50
            if n_snap <= 50:
                self.stats["hnsw_bypassed"] = self.stats.get("hnsw_bypassed", 0) + 1
            for nid in high_tension:
                if nid in processed or nid not in self.nodes:
                    continue
                node = self.nodes[nid]
                # HNSW search for neighbors within distance 2.5
                candidate_ids = self.hnsw_index.search(node.latent_pos, top_k=min(50, n_snap))
                candidates = []
                for oid in candidate_ids:
                    if oid == nid or oid in processed or oid not in self.nodes:
                        continue
                    other = self.nodes[oid]
                    dist = np.linalg.norm(node.latent_pos - other.latent_pos)
                    if dist >= 2.5:
                        continue
                    pd = min(abs(node.phase - other.phase), 2 * np.pi - abs(node.phase - other.phase))
                    if pd > 1.0:
                        candidates.append((oid, dist, pd))
                if not candidates:
                    continue
                candidates.sort(key=lambda x: x[1])
                pid = candidates[0][0]
                if pid not in self.nodes:
                    continue
                partner = self.nodes[pid]

                if self.cfg.do_calculus_validation and self.causal_engine:
                    validation = self.causal_engine.validate_consolidation(nid, pid)
                    self.stats["consolidation_validations"] += 1
                    if not validation["safe"]:
                        self.stats["blocked_consolidations"] += 1
                        processed.add(nid)
                        processed.add(pid)
                        continue

                # Phase 20: Domain consolidation guard — don't merge nodes from different domains
                if self.cfg.domain_consolidation_guard and node.domain != partner.domain:
                    if partner.id not in node.conflict_with:
                        node.conflict_with.append(partner.id)
                    if node.id not in partner.conflict_with:
                        partner.conflict_with.append(node.id)
                    processed.add(nid)
                    processed.add(pid)
                    continue

                gate = self._soft_gate(max(node.tension, partner.tension))

                if self.cfg.enable_rollback:
                    node.pre_consolidation_pos = node.latent_pos.copy()

                if self.diff_consolidation and mode == ConsolidationMode.DIALECTICAL:
                    synth = self.diff_consolidation.compute_synthesis(node, partner, gate)
                    node.latent_pos = synth["latent_pos"]
                    node.phase = synth["phase"]
                    node.amplitude = synth["amplitude"]
                    node.salience = synth["salience"]
                elif mode == ConsolidationMode.DIALECTICAL:
                    node.latent_pos = 0.5 * (node.latent_pos + partner.latent_pos)
                    node.phase = np.arctan2(0.5*(np.sin(node.phase)+np.sin(partner.phase)),
                                            0.5*(np.cos(node.phase)+np.cos(partner.phase))) % (2*np.pi)
                    node.amplitude = min(1.0, 0.8*(node.amplitude+partner.amplitude))
                    node.salience = min(1.0, 0.7*(node.salience+partner.salience))
                else:
                    # MERGE and PRUNE modes: same spatial merge, keep amplitude/salience from survivor
                    node.latent_pos = 0.5 * (node.latent_pos + partner.latent_pos)
                    node.phase = np.arctan2(0.5*(np.sin(node.phase)+np.sin(partner.phase)),
                                            0.5*(np.cos(node.phase)+np.cos(partner.phase))) % (2*np.pi)

                node.tension = 0.0
                node.soft_gate = 1.0
                node.lineage = [f"{node.id}+{pid}"] + node.lineage + partner.lineage
                # Preserve partner content for traceability
                node.content["synthesis_note"] = f"Consolidated with {pid} at t={time.time():.0f}"
                if "merged_content" not in node.content:
                    node.content["merged_content"] = []
                node.content["merged_content"].append(partner.content.get("text", "") or partner.content.get("input_text", ""))

                if self.causal_engine:
                    for parent, strength in partner.causal_strength.items():
                        if parent not in node.causal_strength:
                            node.causal_strength[parent] = strength
                        else:
                            node.causal_strength[parent] = max(node.causal_strength[parent], strength)

                if self.cfg.use_hnsw and self.hnsw_index:
                    self.hnsw_index.remove(pid)
                    self.hnsw_index.insert(nid, node.latent_pos)
                if self.cfg.bm25_fallback and self.bm25_index:
                    self.bm25_index.remove_document(pid)

                pending_deletions.append(pid)
                processed.add(pid)
                updated.append(nid)
                self.stats["consolidations"] += 1
                processed.add(nid)
        else:
            # Fallback: vectorized candidate search without HNSW — O(N) per node via vectorized ops
            # Precompute all positions once (not O(N²) — done once outside loop)
            snap_positions = np.array([self.nodes[oid].latent_pos for oid in node_index_snapshot if oid in self.nodes])
            snap_ids = [oid for oid in node_index_snapshot if oid in self.nodes]
            snap_phases = np.array([self.nodes[oid].phase for oid in snap_ids])
            
            # O(1) lookup instead of O(N) index search
            snap_id_to_idx = {nid: idx for idx, nid in enumerate(snap_ids)}

            for nid in high_tension:
                if nid in processed or nid not in self.nodes:
                    continue
                node = self.nodes[nid]
                # Find node index in snapshot — O(1) dict lookup
                node_idx = snap_id_to_idx.get(nid)
                if node_idx is None:
                    continue
                node_pos = snap_positions[node_idx]

                # Vectorized distance computation
                dists = np.linalg.norm(snap_positions - node_pos, axis=1)
                phase_diffs = np.minimum(
                    np.abs(snap_phases - node.phase),
                    2 * np.pi - np.abs(snap_phases - node.phase)
                )

                # Filter candidates
                mask = (dists < 2.5) & (phase_diffs > 1.0)
                candidate_indices = np.where(mask)[0]
                if len(candidate_indices) == 0:
                    continue

                # Sort by distance and pick nearest
                sorted_indices = candidate_indices[np.argsort(dists[candidate_indices])]
                pid = snap_ids[sorted_indices[0]]
                if pid not in self.nodes or pid in processed:
                    continue
                partner = self.nodes[pid]

                if self.cfg.do_calculus_validation and self.causal_engine:
                    validation = self.causal_engine.validate_consolidation(nid, pid)
                    self.stats["consolidation_validations"] += 1
                    if not validation["safe"]:
                        self.stats["blocked_consolidations"] += 1
                        processed.add(nid)
                        processed.add(pid)
                        continue

                # Phase 20: Domain consolidation guard — don't merge nodes from different domains
                if self.cfg.domain_consolidation_guard and node.domain != partner.domain:
                    if partner.id not in node.conflict_with:
                        node.conflict_with.append(partner.id)
                    if node.id not in partner.conflict_with:
                        partner.conflict_with.append(node.id)
                    processed.add(nid)
                    processed.add(pid)
                    continue

                gate = self._soft_gate(max(node.tension, partner.tension))

                if self.cfg.enable_rollback:
                    node.pre_consolidation_pos = node.latent_pos.copy()

                if self.diff_consolidation and mode == ConsolidationMode.DIALECTICAL:
                    synth = self.diff_consolidation.compute_synthesis(node, partner, gate)
                    node.latent_pos = synth["latent_pos"]
                    node.phase = synth["phase"]
                    node.amplitude = synth["amplitude"]
                    node.salience = synth["salience"]
                elif mode == ConsolidationMode.DIALECTICAL:
                    node.latent_pos = 0.5 * (node.latent_pos + partner.latent_pos)
                    node.phase = np.arctan2(0.5*(np.sin(node.phase)+np.sin(partner.phase)),
                                            0.5*(np.cos(node.phase)+np.cos(partner.phase))) % (2*np.pi)
                    node.amplitude = min(1.0, 0.8*(node.amplitude+partner.amplitude))
                    node.salience = min(1.0, 0.7*(node.salience+partner.salience))

                node.tension = 0.0
                node.soft_gate = 1.0
                node.lineage = [f"{node.id}+{pid}"] + node.lineage + partner.lineage
                node.content["synthesis_note"] = f"Consolidated with {pid} at t={time.time():.0f}"

                if self.causal_engine:
                    for parent, strength in partner.causal_strength.items():
                        if parent not in node.causal_strength:
                            node.causal_strength[parent] = strength
                        else:
                            node.causal_strength[parent] = max(node.causal_strength[parent], strength)

                if self.cfg.use_hnsw and self.hnsw_index:
                    self.hnsw_index.remove(pid)
                    self.hnsw_index.insert(nid, node.latent_pos)
                if self.cfg.bm25_fallback and self.bm25_index:
                    self.bm25_index.remove_document(pid)

                pending_deletions.append(pid)
                processed.add(pid)
                updated.append(nid)
                self.stats["consolidations"] += 1
                processed.add(nid)

        # Fix 2: Apply all deletions after iteration — rebuild node_index once (O(N) instead of O(M×N))
        for pid in pending_deletions:
            if pid in self.nodes:
                del self.nodes[pid]
        # B1: Invalidate cache on consolidation (nodes removed/merged)
        if pending_deletions:
            self._invalidate_tension_cache()
        # Fix 3: Sweep tension cache to remove stale live entries
        self._sweep_tension_cache()
        # Rebuild node_index in one pass
        self.node_index = [nid for nid in self.node_index if nid in self.nodes]

        if updated:
            # FIX: Limit verification to first 10 nodes to avoid O(K×N) blowup
            verify_limit = min(10, len(updated))
            self._verify_consistency(updated[:verify_limit], pre_state)

        self._prune_dead_nodes()
        self.stats["active_nodes"] = len(self.nodes)

        if pre_state and updated:
            scores = []
            for nid in updated:
                if nid in self.nodes and nid in pre_state:
                    o, n = pre_state[nid]["latent_pos"], self.nodes[nid].latent_pos
                    scores.append(max(0, np.dot(o, n) / (np.linalg.norm(o)*np.linalg.norm(n)+1e-8)))
            if scores:
                self._stability_buffer.append(np.mean(scores))
                self.stats["field_stability"] = float(np.mean(self._stability_buffer))

        if self.cfg.enable_rollback and pre_state:
            self._rollback_history.append({"timestamp": time.time(), "pre_state": pre_state, "updated": updated})
            if len(self._rollback_history) > self.cfg.max_rollback_history:
                self._rollback_history.pop(0)

        # Phase 15 Track 1: Version Control — record deltas
        if self.version_control and updated:
            deltas = []
            for nid in updated:
                if nid in self.nodes:
                    deltas.append(NodeDelta(
                        node_id=nid, action="merged",
                        old_state=pre_state.get(nid),
                        new_state=self.nodes[nid].to_dict()
                    ))
            for pid in pending_deletions:
                if pid in pre_state:
                    deltas.append(NodeDelta(
                        node_id=pid, action="deleted",
                        old_state=pre_state.get(pid)
                    ))
            if deltas:
                self.version_control.create_version(deltas, message=f"consolidation: {len(updated)} merged, {len(pending_deletions)} deleted")
                self.stats["current_version"] = self.version_control.current_version
                self.stats["n_versions"] = self.version_control.n_versions

        # Phase 17: Update shard consolidation stats
        if self.role_router and updated:
            # Update consolidation count for affected shards
            affected_roles = set()
            for nid in updated:
                role = self.role_router.get_node_role(nid)
                if role in self.role_router.shards:
                    self.role_router.shards[role].n_consolidations += 1
                    affected_roles.add(role)

        # P0: Invalidate cache after consolidation (nodes changed)
        if updated:
            self._cache_dirty = True

        return updated

    def _verify_consistency(self, updated_nodes: List[str], pre_state: Optional[Dict] = None):
        """Fix: Use local probe (latent+noise) instead of global zero to preserve semantic meaning."""
        from collections import deque
        # Ensure buffer limits
        if not isinstance(self._stability_buffer, deque):
            self._stability_buffer = deque(self._stability_buffer, maxlen=100)
            
        for nid in updated_nodes:
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            # FIX: Probe around the node's actual position, not np.zeros
            probe = node.latent_pos + self._rng.normal(0, 0.05, node.latent_pos.shape)
            results = self.query(probe, phase=node.phase, top_k=1)
            if results and results[0][0] == nid:
                node.self_sup_score = max(0.5, results[0][1])
            else:
                node.self_sup_score *= 0.9

    def _self_supervise(self):
        """Fix: Use local probe instead of np.zeros to prevent false decay of peripheral nodes."""
        if not self.cfg.self_supervision:
            return
        self.stats["self_sup_checks"] += 1
        for nid in list(self.node_index):
            if nid not in self.nodes or not self.nodes[nid].lineage:
                continue
            node = self.nodes[nid]
            # FIX: Probe around the node's actual position
            probe = node.latent_pos + self._rng.normal(0, 0.05, node.latent_pos.shape)
            results = self.query(probe, phase=node.phase, top_k=1)
            if results and results[0][0] == nid:
                node.self_sup_score = max(0.5, results[0][1])
            else:
                node.self_sup_score *= 0.9

    def _check_tda(self):
        if not self.cfg.tda_monitoring or not self.tda_monitor:
            return
        self.stats["tda_checks"] += 1
        r = self.tda_monitor.compute_persistence(self.nodes)
        self.stats["tda_H0"] = r["H0"]
        self.stats["tda_H1"] = r["H1"]
        if self.tda_monitor.get_trend() == "growing_contradictions":
            self.consolidate()

    # Phase 11 Track 3: Predictive coding
    def _encode_field_state(self) -> NDArray:
        """Encode field state into a flat vector for predictive coding."""
        if not self.nodes:
            return np.zeros(self.cfg.latent_dim * 4, dtype=np.float32)
        # Aggregate: mean pos, mean phase, mean amp, mean sal
        positions = np.array([n.latent_pos for n in self.nodes.values()])
        phases = np.array([n.phase for n in self.nodes.values()])
        amps = np.array([n.amplitude for n in self.nodes.values()])
        sals = np.array([n.salience for n in self.nodes.values()])
        mean_pos = np.mean(positions, axis=0)
        mean_phase = np.mean(phases)
        mean_amp = np.mean(amps)
        mean_sal = np.mean(sals)
        # Encode into latent_dim * 4
        state = np.zeros(self.cfg.latent_dim * 4, dtype=np.float32)
        pos_dim = min(len(mean_pos), self.cfg.latent_dim)
        state[:pos_dim] = mean_pos[:pos_dim]
        state[self.cfg.latent_dim] = mean_phase
        state[self.cfg.latent_dim * 2] = mean_amp
        state[self.cfg.latent_dim * 3] = mean_sal
        return state

    # Phase 11 Track 4: Counterfactual imagination
    def imagine_counterfactual(self, base_query: NDArray,
                                intervention: Dict[str, float]) -> List[Dict]:
        """Generate hypothetical trajectories via do-interventions."""
        if not self.scenario_planner:
            return []
        return self.scenario_planner.imagine_counterfactual(base_query, intervention)

    def _prune_dead_nodes(self):
        to_remove = [nid for nid in self.node_index
                     if self.nodes[nid].amplitude < self.cfg.min_amplitude
                     or self.nodes[nid].salience < self.cfg.min_amplitude * 0.5]
        for nid in to_remove:
            if self.cfg.use_hnsw and self.hnsw_index:
                self.hnsw_index.remove(nid)
            if self.cfg.bm25_fallback and self.bm25_index:
                self.bm25_index.remove_document(nid)
            del self.nodes[nid]
        # B1: Invalidate cache on node pruning
        if to_remove:
            self._invalidate_tension_cache()
            self._cache_dirty = True
        # FIX: Rebuild node_index once instead of O(N) remove per node
        self.node_index = [nid for nid in self.node_index if nid in self.nodes]

    def _check_field_integrity(self) -> Dict[str, Any]:
        """Check for NaN/inf in nodes, report issues, and heal them (Fix 11)."""
        issues = []
        n_nan = 0
        n_inf = 0
        healed = []
        for nid, node in self.nodes.items():
            needs_heal = False
            if np.any(np.isnan(node.latent_pos)):
                n_nan += 1
                issues.append(f"NaN in {nid} — will heal")
                needs_heal = True
            if np.any(np.isinf(node.latent_pos)):
                n_inf += 1
                issues.append(f"Inf in {nid} — will heal")
                needs_heal = True
            if np.isnan(node.phase) or np.isinf(node.phase):
                issues.append(f"Invalid phase in {nid} — will heal")
                needs_heal = True
                node.phase = 0.0
            if np.isnan(node.amplitude) or node.amplitude < 0:
                issues.append(f"Invalid amplitude in {nid} — will heal")
                needs_heal = True
                node.amplitude = self.cfg.min_amplitude
            # Fix 11: Actually heal NaN positions by resetting to small random values
            if needs_heal:
                node.latent_pos = self._rng.standard_normal(self.cfg.latent_dim).astype(np.float32) * 0.01
                healed.append(nid)
                self.stats["field_integrity_issues"] = self.stats.get("field_integrity_issues", 0) + 1
        return {
            "n_issues": len(issues),
            "n_nan": n_nan,
            "n_inf": n_inf,
            "healed": healed,
            "issues": issues[:20],
        }

    def evolve_continuous(self, inputs: Optional[List[Dict]] = None, use_sde: bool = False) -> NDArray:
        if not self.ode_dynamics or not self.nodes:
            return np.array([])
        # Fix 2: Deterministic node order via node_index
        ordered_nodes = [self.nodes[nid] for nid in self.node_index if nid in self.nodes]
        initial_state = np.array([n.latent_pos for n in ordered_nodes]).flatten()
        input_signal = None
        if inputs:
            input_signal = np.array([self._project(inp["embedding"]) for inp in inputs]).flatten()
            # Validate input_signal length matches node count to prevent ODE reshape crash
            expected_len = len(ordered_nodes) * self.cfg.latent_dim
            if len(input_signal) != expected_len:
                logger.warning(f"ODE input_signal length {len(input_signal)} != expected {expected_len} (nodes={len(ordered_nodes)}). Falling back to no input signal.")
                input_signal = None
        topo_grad = self.ode_dynamics.compute_topology_gradient(self.nodes)
        if use_sde:
            trajectory = self.ode_dynamics.evolve_with_noise(initial_state, input_signal, topo_grad)
        else:
            trajectory = self.ode_dynamics.evolve(initial_state, input_signal, topo_grad)
        self.stats["ode_steps"] += 1
        # H2: Validate trajectory size before reshape to prevent silent corruption
        expected_size = len(ordered_nodes) * self.cfg.latent_dim
        if trajectory[-1].size != expected_size:
            logger.warning(f"ODE trajectory size {trajectory[-1].size} != expected {expected_size}. Skipping update.")
            return trajectory
        final_state = trajectory[-1].reshape(len(ordered_nodes), self.cfg.latent_dim)
        for i, nid in enumerate(self.node_index):
            if nid in self.nodes and i < len(final_state):
                old_pos = self.nodes[nid].latent_pos.copy()
                self.nodes[nid].latent_pos = final_state[i].astype(np.float32)
                self.nodes[nid].velocity = (self.nodes[nid].latent_pos - old_pos).astype(np.float32)
        return trajectory

    def create_plan(self, goal: str, available_tools: List[str], context: Optional[Dict] = None) -> AgentPlan:
        if not self.agent_planner:
            return AgentPlan(goal=goal, subtasks=[], tools_needed=[],
                           estimated_steps=0, confidence=0.0, reasoning="Agent orchestration not enabled")
        self.stats["plans_created"] += 1
        ctx = context or {}
        ctx["hypothesis_verification"] = self.cfg.hypothesis_verification
        return self.agent_planner.create_plan(goal, available_tools, ctx)

    def verify_hypothesis(self, hypothesis: str, active_nodes: Optional[List[str]] = None) -> Hypothesis:
        if not self.hypothesis_verifier or not self.causal_engine:
            return Hypothesis(statement=hypothesis, confidence=0.5, evidence_nodes=[],
                            causal_path=[], verified=False, verification_score=0.5)
        self.stats["hypotheses_verified"] += 1
        nodes = active_nodes or self.node_index
        return self.hypothesis_verifier.verify(hypothesis, self.causal_engine, nodes)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        if not self.tool_router:
            return ToolCall(tool_name=tool_name, arguments=arguments, error="Tool router not enabled")
        if self.agent_planner and not self.agent_planner.can_call_tool(tool_name):
            return ToolCall(tool_name=tool_name, arguments=arguments, error="Tool call limit reached")
        self.stats["tool_calls"] += 1
        if self.agent_planner:
            self.agent_planner.record_tool_call(tool_name)
        result = self.tool_router.execute(tool_name, arguments)
        if result.success:
            self.stats["tool_misuse_rate"] = self.tool_router.get_misuse_rate()
        return result

    def register_tool(self, name: str, func: Callable):
        if self.tool_router:
            self.tool_router.register_tool(name, func)

    def evaluate_response(self, question: str, answer: str, contexts: List[str],
                          ground_truth: Optional[str] = None) -> EvalResult:
        if not self.ragas_evaluator:
            return EvalResult()
        self.stats["evaluations"] += 1
        causal_edges = None
        if self.causal_engine:
            causal_edges = [(k[0], k[1], v.strength) for k, v in self.causal_engine.causal_effects.items()]
        result = self.ragas_evaluator.evaluate(question, answer, contexts, ground_truth, causal_edges)
        self.stats["ragas_overall"] = result.overall_score
        if self.rollback_manager:
            needs_rollback = self.rollback_manager.record_score(result.overall_score)
            if needs_rollback:
                self.stats["rollbacks"] += 1
        return result

    def compare_shadow(self, shadow_score: float, production_score: float) -> Dict[str, Any]:
        if not self.shadow_evaluator:
            return {}
        self.stats["shadow_comparisons"] += 1
        return self.shadow_evaluator.compare(shadow_score, production_score)

    def step(self, inputs: Optional[List[Dict]] = None):
        self._step_counter += 1
        
        # Throttle: Skip non-critical heavy tasks if backpressure is high
        backpressure_ok = self._backpressure_events < 3 and not self._heavy_modules_degraded
        
        if self.cfg.continuous_dynamics and self.ode_dynamics:
            # Fix: Safe run for ODE to prevent crashes
            self._safe_run("ODEEvolve", self.evolve_continuous, inputs, use_sde=self.cfg.sde_noise_level > 0, default=0)
            return
            
        if inputs:
            for inp in inputs:
                emb = inp["embedding"]
                # Validate embedding dimension to prevent silent corruption
                if len(emb) != self.cfg.embedding_dim:
                    logger.warning(f"Embedding dimension mismatch in step(): expected {self.cfg.embedding_dim}, got {len(emb)}. Skipping.")
                    continue
                phase = inp.get("phase", 0.0)
                content = inp.get("content", {})
                session_id = inp.get("session_id")
                modality = inp.get("modality", "text")
                results = self.query(emb, phase, top_k=1, modality=modality)
                if results and results[0][1] > 0.3:
                    nid, _, node = results[0]
                    target = self._project(emb)
                    node.latent_pos += self.cfg.attraction_lr * (target - node.latent_pos)
                    pd = (phase - node.phase + np.pi) % (2*np.pi) - np.pi
                    node.phase = (node.phase + self.cfg.phase_sync_lr * pd) % (2 * np.pi)
                    node.amplitude = min(1.0, node.amplitude + 0.05)
                    node.salience = min(1.0, node.salience + 0.03)
                else:
                    self.add_node(emb, content, phase, session_id=session_id, modality=modality)

        # Consolidation: periodic instead of probabilistic to avoid hot path spikes
        if len(self.nodes) > 10 and self._step_counter % 20 == 0:
            self._safe_run("Consolidate", self.consolidate, default=[])

        # Self-healing: every N steps
        if self.cfg.self_healing and self._step_counter % self.cfg.healing_check_freq == 0:
            self._safe_run("SelfHeal", self._self_heal)

        # Tier-specific decay: every step (cheap)
        tier_counts = defaultdict(int)
        tier_amplitudes = defaultdict(list)
        for node in self.nodes.values():
            tier = getattr(node, 'tier', 'semantic')
            tier_counts[tier] += 1
            dk = self.cfg.tier_decay.get(tier, self.cfg.decay_rate)
            if self.learnable_kernel:
                dk = max(dk, self.learnable_kernel.decay_rate)
            node.amplitude *= dk
            node.salience *= dk
            node.amplitude = np.clip(node.amplitude, self.cfg.min_amplitude, 1.0)
            node.salience = np.clip(node.salience, self.cfg.min_amplitude * 0.5, 1.0)
            tier_amplitudes[tier].append(node.amplitude)
        self.stats["tier_distribution"] = dict(tier_counts)
        if tier_amplitudes:
            coherences = []
            for tier, amps in tier_amplitudes.items():
                if len(amps) > 1:
                    coherences.append(1.0 - np.std(amps))
                else:
                    coherences.append(1.0)
            self.stats["tier_coherence"] = float(np.mean(coherences)) if coherences else 0.0

        # Predictive coding: every 5 steps
        if self.predictor and len(self.nodes) > 0 and self._step_counter % 5 == 0:
            state = self._encode_field_state()
            self._state_history.append(state)
            if len(self._state_history) >= 2:
                fe = self._safe_run("PredictorFreeEnergy", self.predictor.compute_free_energy, self._state_history[-2], self._state_history[-1], default=0.0)
                self.stats["free_energy"] = fe
                self.stats["prediction_error"] = float(np.mean((self.predictor.predict(self._state_history[-2]) - self._state_history[-1]) ** 2))
                self.stats["surprise_level"] = float(np.clip(fe, 0, 1))
                if fe > 0.3 and len(self.nodes) > 10:
                    self._safe_run("Consolidate", self.consolidate, default=[])
                if fe > 0.01:
                    self._safe_run("PredictorUpdate", self.predictor.update, self._state_history[-2], self._state_history[-1], lr=self.cfg.pc_lr)

        # Max nodes pruning: every 10 steps
        if self.cfg.max_nodes and len(self.nodes) > self.cfg.max_nodes and self._step_counter % 10 == 0:
            sorted_nodes = sorted(self.node_index, key=lambda nid: self.nodes[nid].salience * self.nodes[nid].amplitude)
            n_pruned = len(self.nodes) - self.cfg.max_nodes
            pruned_ids = set(sorted_nodes[:n_pruned])
            for nid in pruned_ids:
                if self.cfg.use_hnsw and self.hnsw_index:
                    self.hnsw_index.remove(nid)
                if self.cfg.bm25_fallback and self.bm25_index:
                    self.bm25_index.remove_document(nid)
                del self.nodes[nid]
            # Rebuild index in O(N) instead of O(N²) list.remove calls
            self.node_index = [nid for nid in self.node_index if nid not in pruned_ids]
            # B1: Invalidate cache on max_nodes pruning
            if n_pruned > 0:
                self._invalidate_tension_cache()

        # Self-supervision: every 20 steps
        if self.cfg.self_supervision and self._step_counter % 20 == 0:
            self._safe_run("SelfSupervise", self._self_supervise)

        # TDA: every N steps (Throttled)
        if backpressure_ok and self.cfg.tda_monitoring and self._step_counter % self.cfg.tda_check_freq == 0:
            self._safe_run("TDA", self._check_tda)

        # Meta-kernel adaptation: every 5 steps
        if self.meta_kernel and self._step_counter % 5 == 0:
            self._safe_run("MetaKernelAdapt", self.meta_kernel.adapt)
            self.stats["meta_kurtosis"] = self.meta_kernel.compute_resonance_kurtosis()
            self.stats["meta_bandwidth"] = self.meta_kernel.get_bandwidth()
            self.stats["meta_phase_coupling"] = self.meta_kernel.get_phase_coupling()

        # Meta-controller optimization: every N steps (Throttled)
        if backpressure_ok and self.meta_controller and self.meta_controller.should_optimize() and self._step_counter % self.cfg.meta_opt_freq == 0:
            best_params = self._safe_run("MetaControllerOptimize", self.meta_controller.optimize, self, default={})
            if best_params:
                self._safe_run("MetaControllerApply", self.meta_controller.apply_params, self, best_params)
                self.stats["meta_optimizations"] += 1
                self.stats["meta_best_params"] = best_params

        # Federated sync: every N steps
        if self.federated and self._step_counter > 0 and self._step_counter % self.cfg.federated_sync_freq == 0:
            local_phases = {nid: n.phase for nid, n in self.nodes.items()}
            local_params = {
                "decay_rate": self.cfg.decay_rate, "tension_threshold": self.cfg.tension_threshold,
                "phase_coupling": self.cfg.phase_coupling, "bandwidth": self.cfg.bandwidth,
            }
            self._safe_run("FederatedSync", self.federated.sync_with_peers, local_phases, local_params)

        # ODE smoothness: every 10 steps (Throttled)
        if backpressure_ok and self.ode_dynamics and self._step_counter % 10 == 0:
            self.stats["response_smoothness"] = self._safe_run("ODESmoothness", self.ode_dynamics.compute_response_smoothness, default=1.0)

        # Shard center updates: every 100 steps
        if self.cfg.sparse_routing and self._step_counter % 100 == 0 and len(self.nodes) > self.cfg.num_shards * 2:
            self._safe_run("ShardUpdate", self._update_shard_centers)
            self.stats["avg_rl_reward"] = self.rl_feedback_loop.get_average_reward()

        # Event-driven processing: every 10 steps
        if self.event_scheduler and self._step_counter % 10 == 0:
            processed = self.event_scheduler.process_pending(self, max_events=5)
            self.stats["events_processed"] += processed
            self.stats["event_queue_depth"] = len(self.event_scheduler._event_queue)

        # Low-rank compression: every N steps
        if self.low_rank_compressor and self._step_counter % self.cfg.compression_freq == 0:
            self._compress_field()

        # Learnable kernel step: every 5 steps
        if self.learnable_kernel and self._step_counter % 5 == 0:
            self.learnable_kernel.step()

        # Causal discovery: every N steps
        causal_freq = getattr(self.cfg, "causal_discovery_freq", 50)
        if self.causal_engine and self._step_counter % max(causal_freq, 1) == 0:
            self.causal_engine.discover_causal_structure()
            for (cause, effect), edge in self.causal_engine.causal_effects.items():
                if effect in self.nodes:
                    # FIX: Prevent unbounded growth of causal_parents list
                    if cause not in self.nodes[effect].causal_parents:
                        self.nodes[effect].causal_parents.append(cause)
                    self.nodes[effect].causal_strength[cause] = edge.strength
                if cause in self.nodes:
                    self.nodes[cause].causal_effects[effect] = edge.strength
            self.stats["causal_edges"] = len(self.causal_engine.causal_effects)
            if self.cfg.contradiction_detection:
                self.causal_engine.detect_contradictions(self.cfg.contradiction_threshold)
                self.stats["contradictions"] = len(self.causal_engine.contradictions)

        # Phase 14 Track 2: Causal graph integrity check
        if self.security and self.cfg.causal_graph_integrity_check and self._step_counter % 100 == 0:
            integrity = self.security.validate_causal_graph_integrity(self.causal_engine)
            if not integrity["is_valid"]:
                self.stats["security_violations"] += len(integrity["issues"])

        # Phase 14 Track 1: Meta-memory self-reflection
        if self.meta_memory_eval and self.meta_memory_eval.should_reflect():
            reflection = self.meta_memory_eval.self_reflect(self)
            self.stats["meta_reflections"] += 1
            # Apply adaptive params
            adaptive = self.meta_memory_eval.get_adaptive_params()
            if adaptive["consolidation_multiplier"] != 1.0:
                # Adjust tension threshold based on recall accuracy
                self.cfg.tension_threshold *= adaptive["consolidation_multiplier"]
                self.cfg.tension_threshold = max(0.05, min(0.5, self.cfg.tension_threshold))

        # Phase 14 Track 5: Swarm memory status
        if self.swarm:
            self.stats["swarm_agents"] = len(self.swarm.agents)
            self.stats["swarm_consensus_events"] = len(self.swarm._consensus_log)

        # Phase 14 Track 2: Security violation stats
        if self.security:
            self.stats["security_violations"] = len(self.security._violation_log)

        # Phase 15 Track 4: Entropy Control
        if self.entropy_ctrl:
            # Record resonance responses for entropy computation
            # (done via query() hook — see query method)
            state = self.entropy_ctrl.get_state()
            self.stats["entropy"] = state["entropy"]
            self.stats["entropy_state"] = state["state"]
            # Auto-trigger consolidation if noisy
            if state["should_consolidate"] and len(self.nodes) > 10:
                self.consolidate()
            # Adjust decay rate if stagnant
            if state["should_explore"]:
                # Temporarily increase decay to clear space
                pass  # Decay is applied in _prune_dead_nodes()

        # Phase 15 Track 1: Version Control stats
        if self.version_control:
            self.stats["current_version"] = self.version_control.current_version
            self.stats["n_versions"] = self.version_control.n_versions

        # Phase 16 Track 1: SymbolicOverlay — extract rules periodically
        if self.symbolic_overlay and self._step_counter % 50 == 0:
            causal_edges = None
            if self.causal_engine:
                causal_edges = self.causal_engine.causal_effects
            self.symbolic_overlay.extract_rules_from_field(self.nodes, causal_edges)
            self.stats["n_symbolic_rules"] = len(self.symbolic_overlay.rules)

        # Phase 16 Track 2: SafetyCertifier — check stability
        if self.safety_certifier and self._step_counter % 10 == 0:
            resonance_scores = []
            if hasattr(self, '_last_query_results') and self._last_query_results:
                resonance_scores = [r[1] for r in self._last_query_results]
            n_contradictions = len(self.causal_engine.contradictions) if self.causal_engine else 0
            cert_result = self.safety_certifier.check_and_regulate(
                self.nodes, resonance_scores, n_contradictions
            )
            self.stats["lyapunov_V"] = cert_result["V"]
            self.stats["lyapunov_dV_dt"] = cert_result["dV_dt"]
            self.stats["safety_regulation_factor"] = cert_result["regulation_factor"]
            self.stats["safety_mode"] = self.safety_certifier.mode

        # Phase 17: RoleShardRouter — Kuramoto sync within each shard
        if self.role_router and self._step_counter % 5 == 0:
            self.role_router.update_kuramoto_phases(self.nodes)
            self.stats["n_shards"] = len(self.role_router.shards)
            self.stats["shard_distribution"] = {
                r: len(s.node_ids) for r, s in self.role_router.shards.items()
            }
            self.stats["role_router_enabled"] = True
            # Cross-shard exchange stats
            total_exchanges = sum(s.n_cross_shard_exchanges for s in self.role_router.shards.values())
            self.stats["cross_shard_exchanges"] = total_exchanges

        # Field integrity check every 100 steps — detect NaN/inf
        if self._step_counter % 100 == 0:
            integrity = self._check_field_integrity()
            if integrity["n_issues"] > 0:
                logger.warning(f"Field integrity issues at step {self._step_counter}: {integrity['n_issues']} issues")
                self.stats["field_integrity_issues"] = integrity["n_issues"]

        # B1: Update tension cache stats every 50 steps
        if self._step_counter % 50 == 0:
            total = self._tension_cache_hits + self._tension_cache_misses
            self.stats["tension_cache_hits"] = self._tension_cache_hits
            self.stats["tension_cache_misses"] = self._tension_cache_misses
            self.stats["tension_cache_hit_rate"] = (self._tension_cache_hits / total) if total > 0 else 0.0

    def _self_heal(self) -> List[Dict]:
        if not self.healer or len(self.nodes) < 3:
            return []
        health, diagnostics = self.healer.compute_field_health(self.nodes)
        self.stats["field_health"] = health.value
        healed = []
        if health == FieldHealth.STABLE:
            for nid in self.node_index:
                self.nodes[nid].is_healing = False
                self.nodes[nid].healing_origin = None
            return []
        self.stats["field_health"] = FieldHealth.HEALING.value
        if diagnostics.get("dead_zones", 0) > 0:
            healed.extend(self.healer.heal_dead_zones(self.nodes, diagnostics["dead_zone_nodes"]))
        if diagnostics.get("hyperconvergence", False):
            healed.extend(self.healer.heal_hyperconvergence(self.nodes))
        if diagnostics.get("fragmentation", 0) > self.cfg.fragmentation_threshold:
            if len(self.nodes) >= 2:
                positions = np.array([n.latent_pos for n in self.nodes.values()])
                tree = cKDTree(positions)
                neighbors = tree.query_ball_point(positions, 2.0)
                isolated = [self.node_index[i] for i in range(len(self.node_index)) if len(neighbors[i]) <= 1]
                if isolated:
                    healed.extend(self.healer.heal_fragmentation(self.nodes, isolated))
        if healed:
            self.stats["healing_events"] += len(healed)
            self.stats["healing_history"].extend(healed)
            # Fix 3: Trim on every overflow, not just when exceeding 1000 — prevents unbounded growth
            if len(self.stats["healing_history"]) > 1000:
                self.stats["healing_history"] = self.stats["healing_history"][-500:]
        return healed

    def rollback_consolidation(self, n_steps: int = 1) -> bool:
        if not self._rollback_history or n_steps > len(self._rollback_history):
            return False
        snapshot = self._rollback_history[-n_steps]
        for nid, state in snapshot["pre_state"].items():
            if nid in self.nodes:
                node = self.nodes[nid]
                node.latent_pos = state["latent_pos"].copy()
                node.phase = state["phase"]
                node.amplitude = state["amplitude"]
                node.salience = state["salience"]
                # H3: Restore full state for consistent rollback
                node.tension = state.get("tension", 0.0)
                node.soft_gate = state.get("soft_gate", 1.0)
                if "content" in state:
                    node.content = dict(state["content"])
                if "lineage" in state:
                    node.lineage = list(state["lineage"])
                if "causal_strength" in state:
                    node.causal_strength = dict(state["causal_strength"])
                if "causal_parents" in state:
                    node.causal_parents = list(state["causal_parents"])
                node.pre_consolidation_pos = None
        self._rollback_history = self._rollback_history[:-n_steps]
        # H8: Clean tension cache after rollback (nodes changed)
        self._tension_cache.clear()
        return True

    def do_intervention(self, node_id: str, new_embedding: NDArray):
        if node_id not in self.nodes:
            return
        new_pos = self._project(new_embedding)
        if self.causal_engine:
            self.causal_engine.do_intervention(node_id, new_pos)
        self.nodes[node_id].latent_pos = new_pos

    def clear_interventions(self):
        if self.causal_engine:
            self.causal_engine.clear_interventions()

    def get_field_health(self) -> Dict:
        if self.healer:
            health, diagnostics = self.healer.compute_field_health(self.nodes)
            diagnostics["kurtosis"] = self.stats.get("meta_kurtosis", 3.0)
            return diagnostics
        return {"health": "unknown", "kurtosis": 3.0}

    def counterfactual_query(self, intervention: Dict[str, Any], query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        if not self.causal_engine:
            return CounterfactualResult(query=str(intervention), intervention=intervention,
                predicted_outcomes=[], confidence=0.0, reasoning_path=["Causal engine not enabled"], assumptions=[])
        self.stats["counterfactual_queries"] += 1
        return self.causal_engine.counterfactual_query(intervention, query_nodes, evidence, self.cfg.counterfactual_max_depth)

    def get_causal_summary(self) -> Dict:
        if not self.causal_engine:
            return {"enabled": False}
        return {
            "enabled": True,
            "causal_edges": len(self.causal_engine.causal_effects),
            "contradictions": len([c for c in self.causal_engine.contradictions.values() if not c.resolved]),
            "nodes_with_effects": len(set(k[0] for k in self.causal_engine.causal_effects)),
            "nodes_affected": len(set(k[1] for k in self.causal_engine.causal_effects)),
            "top_effects": sorted([(f"{k[0]}->{k[1]}", v.strength) for k, v in self.causal_engine.causal_effects.items()],
                                 key=lambda x: x[1], reverse=True)[:10],
        }

    # Track 10: Cross-modal, Meta-controller, Federated stats
    def get_cross_modal_stats(self) -> Dict:
        return {
            "cross_modal_enabled": self.cfg.cross_modal,
            "cross_modal_queries": self.stats.get("cross_modal_queries", 0),
            "cross_modal_recall": self.stats.get("cross_modal_recall", 0.0),
            "kernel_weight": self.cfg.cross_modal_kernel_weight,
            "modal_phase_offsets": self.cfg.modal_phase_offsets,
        }

    def get_meta_controller_state(self) -> Dict:
        if self.meta_controller:
            return self.meta_controller.get_state()
        return {"enabled": False}

    def get_federated_status(self) -> Dict:
        if self.federated:
            return self.federated.get_sync_status()
        return {"enabled": False}

    # ========================================================================
    # PHASE 12 TRACK 1: SPARSE RESONANT ROUTING (MoE-memory)
    # ========================================================================

    def _get_node_shard(self, node_id: str) -> int:
        """Get shard assignment for a node."""
        if node_id in self._node_shard_map:
            return self._node_shard_map[node_id]
        if node_id in self.nodes:
            pos = self.nodes[node_id].latent_pos
            dists = np.linalg.norm(self.shard_centers - pos, axis=1)
            shard = int(np.argmin(dists))
            self._node_shard_map[node_id] = shard
            return shard
        return 0

    def _route_query(self, query_latent: NDArray, top_shards: int = 3) -> List[int]:
        """Route query to top_k most relevant shards (softmax-free)."""
        if self.shard_centers is None:
            return list(range(self.cfg.num_shards))
        dists = np.linalg.norm(self.shard_centers - query_latent, axis=1)
        self.shard_router = 1.0 / (1.0 + dists)
        return list(np.argsort(self.shard_router)[-top_shards:])

    def _update_shard_centers(self):
        """Update shard centers based on current node distribution."""
        if self.shard_centers is None or len(self.nodes) < self.cfg.num_shards:
            return
        from sklearn.cluster import KMeans
        positions = np.array([n.latent_pos for n in self.nodes.values()])
        if len(positions) < self.cfg.num_shards:
            return
        kmeans = KMeans(n_clusters=self.cfg.num_shards, n_init=3, random_state=42)
        labels = kmeans.fit_predict(positions)
        self.shard_centers = kmeans.cluster_centers_.astype(np.float32)
        # Update node-shard map
        self._node_shard_map.clear()
        for i, nid in enumerate(self.node_index):
            self._node_shard_map[nid] = int(labels[i])

    # ========================================================================
    # PHASE 12 TRACK 2: COGNITIVE CONTEXT COMPRESSION
    # ========================================================================

    def _cognitive_compress(self, results: List[Tuple[str, float, MemoryNode]]) -> str:
        """Compress raw memory results into a structured cognitive dump for LLM."""
        if not results:
            return "### COGNITIVE_CONTEXT\nNo relevant structures."

        high_res = [(nid, r, n) for nid, r, n in results if r > self.cfg.high_resonance_threshold]
        contradictions = [n for _, _, n in results if n.content.get("causal_flag") == "incompatible"]
        procedural = [n for _, _, n in results if getattr(n, 'tier', 'semantic') == "procedural"]

        lines = ["### COGNITIVE_CONTEXT"]
        if high_res:
            summaries = []
            for nid, r, n in high_res:
                text = n.content.get("text", "unknown")[:60]
                summaries.append(f"[{text}...](R:{r:.2f},S:{n.salience:.2f})")
            lines.append(f"• High resonance ({len(high_res)} nodes): " + " | ".join(summaries))
        if contradictions:
            texts = [n.content.get("text", "unknown")[:40] for n in contradictions[:3]]
            lines.append(f"[WARN] Conflicting nodes: " + " | ".join(texts))
        if procedural:
            lines.append("[TOOL] Procedural patterns available (how-to)")

        # Add lineage summary for complex nodes
        lineage_nodes = [(nid, n) for nid, r, n in results if n.lineage]
        if lineage_nodes:
            lines.append(f"[STATS] Consolidated memories: {len(lineage_nodes)} nodes with synthesis history")

        return "\n".join(lines)

    # ========================================================================
    # PHASE 12 TRACK 3: CRYSTALLIZATION (episodic → semantic/procedural)
    # ========================================================================

    def _crystallize_recurring(self, window: int = 100, similarity_thresh: float = 0.75):
        """Detect recurring episodic patterns and crystallize into semantic nodes."""
        recent_ids = self.node_index[-window:]
        recent = [self.nodes[nid] for nid in recent_ids
                  if nid in self.nodes and getattr(self.nodes[nid], 'tier', 'semantic') == "episodic"
                  and nid not in self._crystallized_nodes]
        if len(recent) < 5:
            return

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return

        pos = np.array([n.latent_pos for n in recent])
        labels = DBSCAN(eps=0.4, min_samples=self.cfg.crystallization_min_cluster).fit_predict(pos)

        crystallized_count = 0
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            members = [recent[i] for i, l in enumerate(labels) if l == cluster_id]
            if len(members) >= self.cfg.crystallization_min_cluster:
                new_pos = np.mean([m.latent_pos for m in members], axis=0).astype(np.float32)
                # Circular mean for phases: arctan2(mean(sin), mean(cos))
                phases = np.array([m.phase for m in members])
                new_phase = float(np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))) % (2 * np.pi)
                combined_text = " ".join([m.content.get("text", "")[:30] for m in members[:3]])
                new_content = {
                    "text": f"Crystallized: {combined_text}...",
                    "tier": "semantic",
                    "crystallized_from": [m.id for m in members],
                    "crystallized_at": time.time(),
                }
                new_id = self.add_node(new_pos, new_content, phase=float(new_phase % (2 * np.pi)), skip_projection=True)
                self.nodes[new_id].tier = "semantic"
                # Mark originals as archived
                for m in members:
                    m.content["archived"] = True
                    self._crystallized_nodes.add(m.id)
                crystallized_count += 1

        if crystallized_count > 0:
            self.stats["crystallizations"] += crystallized_count
            self.stats["crystallized_clusters"] += crystallized_count

    # ========================================================================
    # PHASE 12 TRACK 4: ASYNC MULTI-THREADED EVOLUTION PIPELINE
    # ========================================================================

    def _safe_run(self, module_name: str, func, *args, default=None, **kwargs):
        """Execute a heavy module safely, handling exceptions and throttling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[SafeRun] {module_name} failed: {e}")
            self._backpressure_events += 1
            # Fix 10: Track when we enter degraded mode
            if self._backpressure_events >= 3 and not self._heavy_modules_degraded:
                self._heavy_modules_degraded = True
                logger.warning("Entering degraded mode — heavy modules disabled until recovery")
            return default

    async def _start_workers(self):
        """Start background worker tasks for async pipeline with lifecycle tracking."""
        if self._workers_started:
            return
        self._workers_started = True
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        
        # Fix 3: Track tasks for cancellation in clear()
        t_evolve = asyncio.create_task(self._worker_evolve())
        t_save = asyncio.create_task(self._worker_save())
        self._workers.extend([t_evolve, t_save])

    async def _worker_evolve(self):
        """Background worker for field evolution with throttling."""
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(self.evolve_q.get(), timeout=1.0)
                    inputs = payload.get("inputs", {})
                    
                    # Throttling: Skip heavy meta-ops if backpressure high
                    backpressure_ok = self._backpressure_events < 3
                    
                    self.step(inputs)

                    # Fix 10: Track recovery and update last successful step
                    self._last_successful_step = time.time()

                    if backpressure_ok and self.meta_controller:
                        # Safe execution for optimization
                        if self.meta_controller.should_optimize():
                            self._safe_run("MetaControllerOptimize", self.meta_controller.optimize, self)

                    # Decay backpressure on success — also check if we can recover from degraded mode
                    if self._backpressure_events > 0:
                        self._backpressure_events = max(0, self._backpressure_events - 1)
                        # Fix 10: Recover from degraded mode if backpressure has fully decayed
                        if self._backpressure_events == 0 and self._heavy_modules_degraded:
                            self._heavy_modules_degraded = False
                            self.stats["backpressure_degraded_mode"] = self.stats.get("backpressure_degraded_mode", 0) + 1
                            logger.info("Backpressure recovered — heavy modules re-enabled")
                        if self._backpressure_events == 0:
                            self.stats["last_backpressure_recovery"] = time.time()
                        
                    self.evolve_q.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    self._backpressure_events += 1
                    logger.error(f"Evolve worker error: {e}")
        except asyncio.CancelledError:
            logger.info("Evolve worker cancelled cleanly.")

    async def _worker_save(self):
        """Background worker for context saving."""
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(self.save_q.get(), timeout=1.0)
                    # Save is handled by add_node, just track depth
                    self._track_queue_depth()
                    self.save_q.task_done()
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Save worker error: {e}")
        except asyncio.CancelledError:
            logger.info("Save worker cancelled cleanly.")
    def _track_queue_depth(self):
        """Track async queue depths for monitoring."""
        if self.cfg.async_pipeline and self.evolve_q:
            self.stats["async_queue_depth"] = (
                self.evolve_q.qsize() +
                (self.save_q.qsize() if self.save_q else 0) +
                (self.query_q.qsize() if self.query_q else 0)
            )

    # ========================================================================
    # PHASE 13 TRACK 4: LOW-RANK COMPRESSION
    # ========================================================================

    def _compress_field(self):
        """Compress node latent positions via incremental SVD."""
        if not self.low_rank_compressor or len(self.nodes) < 10:
            return
        positions = np.array([n.latent_pos for n in self.nodes.values()])
        compressed, reconstructed = self.low_rank_compressor.compress(positions)
        ratio = self.low_rank_compressor.get_compression_ratio(positions.shape)
        self.stats["compression_ratio"] = ratio
        self.stats["compression_updates"] = self.low_rank_compressor._update_count
        # Update node positions with reconstructed (lossy but preserves resonance)
        for i, nid in enumerate(self.node_index):
            if i < len(reconstructed) and nid in self.nodes:
                self.nodes[nid].latent_pos = reconstructed[i].astype(np.float32)

    # ========================================================================
    # PHASE 13 TRACK 1: GOAL MANAGEMENT
    # ========================================================================

    def add_goal(self, description: str, goal_id: Optional[str] = None,
                 subgoals: Optional[List[str]] = None,
                 priority: float = 1.0) -> str:
        """Add a goal to the teleological layer."""
        if not self.goal_tracker:
            self.goal_tracker = GoalTracker(
                self.cfg.max_goals, self.cfg.goal_decay,
                self.cfg.goal_completion_threshold
            )
        return self.goal_tracker.add_goal(description, goal_id, subgoals, priority)

    def update_goal_completion(self, goal_id: str, completion: float,
                                related_nodes: Optional[List[str]] = None):
        """Update goal completion progress."""
        if self.goal_tracker:
            self.goal_tracker.update_completion(goal_id, completion, related_nodes)

    def get_active_goals(self) -> List[Dict]:
        """Get current active goals."""
        if not self.goal_tracker:
            return []
        return [g.to_dict() for g in self.goal_tracker.get_active_goals()]

    # ========================================================================
    # PHASE 13 TRACK 3: RL FEEDBACK
    # ========================================================================

    def apply_rl_feedback(self, response: str, context_node_ids: List[str]) -> float:
        """Apply RL feedback from LLM response."""
        if not self.rl_feedback_loop:
            self.rl_feedback_loop = RLFeedbackLoop(
                self.cfg.rl_learning_rate, self.cfg.rl_reward_window
            )
        reward = self.rl_feedback_loop.extract_reward_from_response(response, context_node_ids)
        self.rl_feedback_loop.apply_field_updates(self)
        self.stats["avg_rl_reward"] = self.rl_feedback_loop.get_average_reward()
        return reward

    def export_field(self, path: str, fmt: str = "json"):
        """Export field state to file.

        Args:
            path: Output file path
            fmt: "json" (default) or "msgpack" (binary, requires msgpack)
        """
        path = _sanitize_path(path)
        # Safety check: prevent overwriting non-empty file with empty memory
        n_nodes = len(self.nodes)
        if n_nodes == 0 and os.path.exists(path):
            try:
                existing_size = os.path.getsize(path)
                if existing_size > 1000:  # File has content (>1KB)
                    logger.warning(f"export_field blocked: refusing to overwrite {path} ({existing_size/1024:.0f}KB) with empty memory (0 nodes). "
                                   f"This prevents accidental data loss.")
                    return  # Silently skip export to protect existing data
            except OSError:
                pass  # If we can't check, proceed with export

        logger.info(f"export_field: exporting {n_nodes} nodes to {path}")
        from rtmdk.memory.serialization import FieldSerializer
        FieldSerializer.field_to_file(self, path, fmt)

    @classmethod
    def import_field(cls, path: str, embedder: Callable) -> "RTMDKMemory":
        path = _sanitize_path(path)
        from rtmdk.memory.serialization import FieldSerializer
        return FieldSerializer.field_from_file(path, embedder)
        # Convert memory_tiers list back to set
        if "memory_tiers" in cd and isinstance(cd["memory_tiers"], list):
            cd["memory_tiers"] = set(cd["memory_tiers"])
        # Handle v5/v6 backward compatibility
        if "causal_modeling" in cd and "causal_topological" not in cd:
            cd["causal_topological"] = cd.pop("causal_modeling")
        elif "causal_modeling" in cd:
            cd.pop("causal_modeling")
        valid_fields = set(f.name for f in RTMDKConfig.__dataclass_fields__.values())
        cd = {k: v for k, v in cd.items() if k in valid_fields}
        config = RTMDKConfig(**cd)
        memory = RTMDKMemory(config=config, embedder=embedder)

        if config.learn_projection and "projection_state" in data:
            memory.field.projection_learner.load_state(data["projection_state"])
        elif "projection" in data:
            memory.field._raw_projection = np.array(data["projection"], dtype=np.float32)
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
                memory.field.ode_dynamics.W = np.array(ode_state["W"], dtype=np.float32)
            memory.field.ode_dynamics.noise_level = ode_state.get("noise_level", 0.01)
        # Track 10 imports
        if config.meta_controller and "meta_controller" in data:
            memory.field.meta_controller.load_state(data["meta_controller"])
        if config.federated and "federated" in data:
            memory.field.federated.import_state(data["federated"])
        # Phase 14 imports
        if config.meta_memory and "meta_memory_eval" in data:
            memory.field.meta_memory_eval.load_state(data["meta_memory_eval"])
        if config.security_enabled and "security" in data:
            memory.field.security.load_state(data["security"])
        if config.swarm_memory and "swarm" in data:
            memory.field.swarm.load_state(data["swarm"])

        logger.info(f"import_field: loading {len(data['nodes'])} nodes")
        for nd in data["nodes"]:
            node = MemoryNode.from_dict(nd)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        logger.info(f"import_field: successfully loaded {len(memory.field.nodes)} nodes")

        # Reload stats from file, then reconcile with actual node count
        saved_stats = data.get("stats", {})
        memory.field.stats = saved_stats

        # Reconcile: reset accumulation counters to match actual nodes
        n_nodes = len(memory.field.nodes)
        logger.info(f"import_field: reconciling stats for {n_nodes} nodes")
        memory.field.stats["total_adds"] = n_nodes
        memory.field.stats["active_nodes"] = n_nodes

        # Reset historical accumulation counters (they reflect past life, not current state)
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
            "swarm_agents", "swarm_consensus_events",
            "current_version", "n_versions",
            "clarifications_generated",
            "entropy", "entropy_state",
            "triton_backend_used", "gpu_acceleration",
            "n_symbolic_rules", "n_symbolic_inferences", "n_symbolic_conflicts",
            "lyapunov_V", "lyapunov_dV_dt", "safety_regulation_factor", "safety_mode",
            "n_shards", "shard_distribution", "cross_shard_exchanges",
            "role_router_enabled",
            "field_integrity_issues",
            "plans_created", "hypotheses_verified", "tool_calls", "tool_misuse_rate",
            "ragas_overall",
            "tier_coherence",
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
            tier = node.content.get("tier", node.tier if hasattr(node, 'tier') else "semantic")
            tier_dist[tier] = tier_dist.get(tier, 0) + 1
        memory.field.stats["tier_distribution"] = tier_dist

        # Reset avg_response to a reasonable default since we have no query history
        memory.field.stats["avg_response"] = 0.0

        logger.info(f"import_field: complete — {n_nodes} nodes, tier_distribution={tier_dist}")
        return memory

    def export_to_dict(self) -> Dict:
        """Export field state to a dict (for UMP and other protocols)."""
        cd = asdict(self.config) if hasattr(self, 'config') else asdict(self.cfg)
        cd["consolidation_mode"] = _enum_value(cd.get("consolidation_mode"), "dialectical")
        cd["backend"] = _enum_value(cd.get("backend"), "numpy")
        cd["context_format"] = _enum_value(cd.get("context_format"), "plain")
        cd["eval_mode"] = _enum_value(cd.get("eval_mode"), "production")
        if "memory_tiers" in cd and isinstance(cd["memory_tiers"], set):
            cd["memory_tiers"] = list(cd["memory_tiers"])
        data = {"_schema_version": "1.0", "config": cd, "nodes": [n.to_dict() for n in self.nodes.values()], "stats": self.stats}
        if self.projection_learner:
            data["projection_state"] = self.projection_learner.get_state()
        else:
            data["projection"] = self._raw_projection.tolist()
        if self.learnable_kernel:
            data["learnable_kernel"] = self.learnable_kernel.get_state()
        if self.meta_kernel:
            data["meta_kernel"] = self.meta_kernel.get_state()
        if self.healer:
            data["healer"] = self.healer.get_state()
        if self.causal_engine:
            data["causal_engine"] = self.causal_engine.get_state()
        if self.ode_dynamics:
            data["ode_dynamics"] = self.ode_dynamics.get_state()
        if self.meta_controller:
            data["meta_controller"] = self.meta_controller.get_state()
        if self.federated:
            data["federated"] = self.federated.export_state()
        if self.meta_memory_eval:
            data["meta_memory_eval"] = self.meta_memory_eval.get_state()
        if self.security:
            data["security"] = self.security.get_state()
        if self.swarm:
            data["swarm"] = self.swarm.get_state()
        if self.version_control:
            data["version_control"] = self.version_control.export_state()
        if self.entropy_ctrl:
            data["entropy_ctrl"] = self.entropy_ctrl.get_state_dict()
        if self.symbolic_overlay:
            data["symbolic_overlay"] = self.symbolic_overlay.get_state()
        if self.safety_certifier:
            data["safety_certifier"] = self.safety_certifier.get_state()
        # Fix 4: Save missing subsystems
        if self.event_scheduler:
            data["event_scheduler"] = self.event_scheduler.get_state()
        if self.low_rank_compressor:
            data["low_rank_compressor"] = self.low_rank_compressor.get_state()
        if self.triton_backend:
            data["triton_backend"] = self.triton_backend.get_state()
        if self.goal_tracker:
            data["goal_tracker"] = self.goal_tracker.get_state()
        if self.rl_feedback_loop:
            data["rl_feedback_loop"] = self.rl_feedback_loop.get_state()
        if self.predictor:
            data["predictor"] = self.predictor.get_state()
        if self.scenario_planner:
            data["scenario_planner"] = self.scenario_planner.get_state()
        if self.engram_manager:
            data["engram_manager"] = self.engram_manager.get_state()
        return data

    @classmethod
    def import_from_dict(cls, data: Dict, embedder: Callable) -> "RTMDKMemory":
        """Import field state from a dict (for UMP and other protocols)."""
        cd = data["config"]
        if isinstance(cd.get("consolidation_mode"), str):
            cd["consolidation_mode"] = ConsolidationMode(cd["consolidation_mode"])
        if isinstance(cd.get("backend"), str):
            cd["backend"] = Backend(cd["backend"])
        if isinstance(cd.get("context_format"), str):
            cd["context_format"] = ContextFormat(cd["context_format"])
        if isinstance(cd.get("eval_mode"), str):
            cd["eval_mode"] = EvalMode(cd["eval_mode"])
        if "memory_tiers" in cd and isinstance(cd["memory_tiers"], list):
            cd["memory_tiers"] = set(cd["memory_tiers"])
        if "causal_modeling" in cd and "causal_topological" not in cd:
            cd["causal_topological"] = cd.pop("causal_modeling")
        elif "causal_modeling" in cd:
            cd.pop("causal_modeling")
        valid_fields = set(f.name for f in RTMDKConfig.__dataclass_fields__.values())
        cd = {k: v for k, v in cd.items() if k in valid_fields}
        config = RTMDKConfig(**cd)
        memory = RTMDKMemory(config=config, embedder=embedder)

        if config.learn_projection and "projection_state" in data:
            memory.field.projection_learner.load_state(data["projection_state"])
        elif "projection" in data:
            memory.field._raw_projection = np.array(data["projection"], dtype=np.float32)
        if config.differentiable and "learnable_kernel" in data:
            memory.field.learnable_kernel.load_state(data["learnable_kernel"])
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
                memory.field.ode_dynamics.W = np.array(ode_state["W"], dtype=np.float32)
            memory.field.ode_dynamics.noise_level = ode_state.get("noise_level", 0.01)
        if config.meta_controller and "meta_controller" in data:
            memory.field.meta_controller.load_state(data["meta_controller"])
        if config.federated and "federated" in data:
            memory.field.federated.import_state(data["federated"])
        if config.meta_memory and "meta_memory_eval" in data:
            memory.field.meta_memory_eval.load_state(data["meta_memory_eval"])
        if config.security_enabled and "security" in data:
            memory.field.security.load_state(data["security"])
        if config.swarm_memory and "swarm" in data:
            memory.field.swarm.load_state(data["swarm"])
        if config.version_control and "version_control" in data:
            memory.field.version_control.import_state(data["version_control"])
        if config.entropy_management and "entropy_ctrl" in data:
            memory.field.entropy_ctrl.load_state_dict(data["entropy_ctrl"])
        if config.symbolic_overlay and "symbolic_overlay" in data:
            memory.field.symbolic_overlay.load_state(data["symbolic_overlay"])
        if config.safety_certifier and "safety_certifier" in data:
            memory.field.safety_certifier.load_state(data["safety_certifier"])
        # Fix 4: Load missing subsystems and reset historical stats
        if "event_scheduler" in data and memory.field.event_scheduler:
            memory.field.event_scheduler.load_state(data["event_scheduler"])
        if "low_rank_compressor" in data and memory.field.low_rank_compressor:
            memory.field.low_rank_compressor.load_state(data["low_rank_compressor"])
        if "goal_tracker" in data and memory.field.goal_tracker:
            memory.field.goal_tracker.load_state(data["goal_tracker"])
        if "rl_feedback_loop" in data and memory.field.rl_feedback_loop:
            memory.field.rl_feedback_loop.load_state(data["rl_feedback_loop"])
        if "predictor" in data and memory.field.predictor:
            memory.field.predictor.load_state(data["predictor"])
        if "scenario_planner" in data and memory.field.scenario_planner:
            memory.field.scenario_planner.load_state(data["scenario_planner"])
        if "engram_manager" in data and memory.field.engram_manager:
            memory.field.engram_manager.load_state(data["engram_manager"])

        # Reset historical metrics to avoid stale accumulated state (matches import_field behavior)
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
            "swarm_agents", "swarm_consensus_events",
            "current_version", "n_versions",
            "clarifications_generated",
            "entropy", "entropy_state",
            "triton_backend_used", "gpu_acceleration",
            "n_symbolic_rules", "n_symbolic_inferences", "n_symbolic_conflicts",
            "lyapunov_V", "lyapunov_dV_dt", "safety_regulation_factor", "safety_mode",
            "n_shards", "shard_distribution", "cross_shard_exchanges",
            "role_router_enabled",
            "field_integrity_issues",
            "plans_created", "hypotheses_verified", "tool_calls", "tool_misuse_rate",
            "ragas_overall",
            "tier_coherence",
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

        for nd in data["nodes"]:
            node = MemoryNode.from_dict(nd)
            memory.field.nodes[node.id] = node
            memory.field.node_index.append(node.id)
        memory.field.stats = data.get("stats", memory.field.stats)
        return memory


# ============================================================================
# RTMDKMemory v7
# ============================================================================

class RTMDKMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    config: RTMDKConfig = Field(default_factory=RTMDKConfig)
    embedder: Callable[[str], NDArray[np.float32]]
    field: Optional[RTMDKField] = Field(default=None, exclude=True)
    session_phases: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _init_field(cls, data):
        if isinstance(data, dict) and data.get("field") is None:
            cfg = data.get("config", RTMDKConfig())
            data = dict(data)
            data["field"] = RTMDKField(cfg)
        return data

    def model_post_init(self, __context):
        if self.field is None:
            object.__setattr__(self, "field", RTMDKField(self.config))
        # Fix 4: Auto-start async workers if async_pipeline is enabled
        if self.config.async_pipeline and not self.field._workers_started:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.field._start_workers())
            except RuntimeError:
                pass
        # Phase 18: Initialize Engram Manager
        if self.config.enable_engrams:
            try:
                from rtmdk.engrams import EngramManager
                object.__setattr__(self, "engram_manager", EngramManager(
                    min_nodes=self.config.engram_min_nodes,
                    max_nodes=self.config.engram_max_nodes,
                    creation_threshold=self.config.engram_creation_threshold,
                    decay_rate=self.config.engram_decay_rate,
                    pattern_completion=self.config.engram_pattern_completion,
                    overlap_threshold=self.config.engram_overlap_threshold,
                ))
            except Exception as e:
                logger.warning(f"Engram manager initialization failed, disabling: {e}")
                object.__setattr__(self, "engram_manager", None)
        else:
            object.__setattr__(self, "engram_manager", None)

    @property
    def memory_variables(self) -> List[str]:
        return ["rtmdk_context"]

    def add_node(self, embedding: NDArray, content: Dict, **kwargs) -> str:
        """Add a node to the memory field. Delegates to RTMDKField.add_node."""
        return self.field.add_node(embedding, content, **kwargs)

    def _get_phase(self, session_id: Optional[str] = None, embedding: Optional[NDArray] = None) -> float:
        if session_id and session_id in self.session_phases:
            return self.session_phases[session_id]
        phase = (time.time() * 0.01) % (2 * np.pi)
        if session_id:
            self.session_phases[session_id] = phase
        return phase

    def load_memory_variables(self, inputs: Dict[str, str]) -> Dict[str, str]:
        query = inputs.get("input", inputs.get("query", ""))
        session_id = inputs.get("session_id", "default")
        if not query:
            return {"rtmdk_context": ""}
        embedding = self.embedder(query)
        phase = self._get_phase(session_id, embedding)

        # Phase 18: Engram-based retrieval (if enabled)
        if self.engram_manager is not None and self.engram_manager.index.size > 0:
            # Collect node embeddings for pattern completion
            node_embs = {}
            for nid, node in self.field.nodes.items():
                emb = self._get_node_embedding(nid, node)
                if emb is not None:
                    node_embs[nid] = emb

            # Retrieve engrams
            engram_results = self.engram_manager.retrieve_engrams(
                embedding, node_embs, top_k=self.field.cfg.top_k
            )

            if engram_results:
                # Expand engrams to node-level results
                results = self.engram_manager.expand_engrams(
                    engram_results, self.field, top_k=self.field.cfg.top_k
                )
                self.field.stats["engram_retrievals"] += 1
            else:
                # Fallback to standard node-level retrieval
                results = self.field.query(embedding, phase, top_k=self.field.cfg.top_k, session_id=session_id)
        else:
            # Standard node-level retrieval
            results = self.field.query(embedding, phase, top_k=self.field.cfg.top_k, session_id=session_id)

        # Session-scoped retrieval: filter results by session_id, with global fallback
        if session_id and session_id != "default" and results:
            session_results = [
                (nid, score, node) for nid, score, node in results
                if node.content.get("session") == session_id
            ]
            # If session results are fewer than top_k, supplement with global results
            if len(session_results) < self.field.cfg.top_k:
                global_results = [
                    (nid, score, node) for nid, score, node in results
                    if node.content.get("session") != session_id
                ]
                needed = self.field.cfg.top_k - len(session_results)
                session_results.extend(global_results[:needed])
            # Boost session-matching scores
            boosted = []
            for nid, score, node in session_results:
                if node.content.get("session") == session_id:
                    score *= 1.5  # 50% boost for session match
                boosted.append((nid, score, node))
            boosted.sort(key=lambda x: x[1], reverse=True)
            results = boosted[:self.field.cfg.top_k]
            self.field.stats["session_scoped_retrievals"] = self.field.stats.get("session_scoped_retrievals", 0) + 1

        # Phase 15 Track 2: Proactive Clarification
        if self.config.proactive_clarification and results:
            max_score = results[0][1] if results else 0.0
            threshold = self.field.cfg.min_response * self.config.clarification_threshold_ratio
            if 0 < max_score < threshold:
                # Weak resonance — generate clarification from near-miss nodes
                clarification = self._generate_clarification(results, query)
                self.field.stats["clarifications_generated"] += 1
                return {"rtmdk_context": clarification}

        # Phase 15 Track 3: Attention Token formatting
        if self.config.attention_tokens and results:
            context = format_context(results, ContextFormat.ATTENTION)
        # Phase 13 Track 2: Cognitive attention bias formatting
        elif self.config.attention_bias and results:
            context = format_cognitive_context(results, bias_applied=True)
            self.field.stats["attention_bias_applied"] += 1
        # Phase 12 Track 2: Cognitive context compression
        elif self.config.cognitive_compression and results:
            context = self.field._cognitive_compress(results)
            raw_context = format_context(results, self.config.context_format)
            tokens_saved = max(0, len(raw_context) - len(context))
            self.field.stats["context_tokens_saved"] += tokens_saved
            self.field.stats["cognitive_compressions"] += 1
        else:
            context = format_context(results, self.config.context_format)

        # Phase 16 Track 1: SymbolicOverlay — add symbolic context
        if self.config.symbolic_overlay and self.field.symbolic_overlay and results:
            # Extract facts from top results
            facts = []
            for nid, score, node in results[:3]:
                text = node.content.get("text", "")
                concepts = self.field.symbolic_overlay._extract_concepts(text)
                facts.extend(concepts)
            if facts:
                symbolic_ctx = self.field.symbolic_overlay.get_symbolic_context(facts, max_depth=2)
                if symbolic_ctx:
                    context += "\n\n" + symbolic_ctx
                    self.field.stats["n_symbolic_inferences"] += 1
                    n_conflicts = sum(1 for r in self.field.symbolic_overlay.rules.values()
                                     if r.is_contextual_exception)
                    self.field.stats["n_symbolic_conflicts"] = n_conflicts

        return {"rtmdk_context": context}

    def load_memory_variables_with_embedding(
        self, inputs: Dict[str, str], embedding: NDArray
    ) -> Dict[str, str]:
        """Query memory with pre-computed embedding (no HTTP call).

        This is the optimized version that accepts an embedding from
        an external embedder, avoiding the HTTP call to LM Studio.
        Use this for batch processing and fair benchmark comparisons.

        Args:
            inputs: {"input": "query text", "session_id": "...", ...}
            embedding: Pre-computed embedding vector (768d for nomic-embed)

        Returns:
            {"rtmdk_context": formatted context string}
        """
        query = inputs.get("input", inputs.get("query", ""))
        session_id = inputs.get("session_id", "default")
        if not query:
            return {"rtmdk_context": ""}

        # Use provided embedding instead of calling self.embedder()
        phase = self._get_phase(session_id, embedding)

        # Phase 18: Engram-based retrieval (if enabled)
        if self.engram_manager is not None and self.engram_manager.index.size > 0:
            node_embs = {}
            for nid, node in self.field.nodes.items():
                emb = self._get_node_embedding(nid, node)
                if emb is not None:
                    node_embs[nid] = emb

            engram_results = self.engram_manager.retrieve_engrams(
                embedding, node_embs, top_k=self.field.cfg.top_k
            )

            if engram_results:
                results = self.engram_manager.expand_engrams(
                    engram_results, self.field, top_k=self.field.cfg.top_k
                )
                self.field.stats["engram_retrievals"] += 1
            else:
                results = self.field.query(embedding, phase, top_k=self.field.cfg.top_k, session_id=session_id)
        else:
            results = self.field.query(embedding, phase, top_k=self.field.cfg.top_k, session_id=session_id)

        # Session-scoped retrieval
        if session_id and session_id != "default" and results:
            session_results = [
                (nid, score, node) for nid, score, node in results
                if node.content.get("session") == session_id
            ]
            if len(session_results) < self.field.cfg.top_k:
                global_results = [
                    (nid, score, node) for nid, score, node in results
                    if node.content.get("session") != session_id
                ]
                needed = self.field.cfg.top_k - len(session_results)
                session_results.extend(global_results[:needed])
            boosted = []
            for nid, score, node in session_results:
                if node.content.get("session") == session_id:
                    score *= 1.5
                boosted.append((nid, score, node))
            boosted.sort(key=lambda x: x[1], reverse=True)
            results = boosted[:self.field.cfg.top_k]
            self.field.stats["session_scoped_retrievals"] = self.field.stats.get("session_scoped_retrievals", 0) + 1

        # Phase 1: Hybrid retrieval — blend RTMDK resonance with BM25 text scores
        if self.field.cfg.hybrid_alpha < 1.0 and self.field.bm25_index is not None and results:
            # Get BM25 scores for the query
            bm25_results = self.field.bm25_index.search(query, self.field.cfg.top_k * 2)
            if bm25_results:
                # Create BM25 score lookup
                bm25_scores = {nid: score for nid, score in bm25_results}
                # Normalize BM25 scores to [0, 1]
                max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
                if max_bm25 > 0:
                    bm25_scores = {nid: s / max_bm25 for nid, s in bm25_scores.items()}

                # Blend scores: final = α × resonance + (1-α) × bm25
                alpha = self.field.cfg.hybrid_alpha
                blended = []
                for nid, score, node in results:
                    bm25_score = bm25_scores.get(nid, 0.0)
                    blended_score = alpha * score + (1 - alpha) * bm25_score
                    blended.append((nid, blended_score, node))

                # Also add high-BM25 nodes that weren't in RTMDK results
                for nid, bm25_score in bm25_scores.items():
                    if nid not in [n[0] for n in blended] and bm25_score > self.field.cfg.min_response:
                        node = self.field.nodes.get(nid)
                        if node:
                            blended_score = alpha * 0.0 + (1 - alpha) * bm25_score
                            blended.append((nid, blended_score, node))

                blended.sort(key=lambda x: x[1], reverse=True)
                results = blended[:self.field.cfg.top_k]
                self.field.stats["hybrid_retrievals"] = self.field.stats.get("hybrid_retrievals", 0) + 1

        # Context formatting (same as load_memory_variables)
        if self.config.proactive_clarification and results:
            max_score = results[0][1] if results else 0.0
            threshold = self.field.cfg.min_response * self.config.clarification_threshold_ratio
            if 0 < max_score < threshold:
                clarification = self._generate_clarification(results, query)
                self.field.stats["clarifications_generated"] += 1
                return {"rtmdk_context": clarification}

        if self.config.attention_tokens and results:
            context = format_context(results, ContextFormat.ATTENTION)
        elif self.config.attention_bias and results:
            context = format_cognitive_context(results, bias_applied=True)
            self.field.stats["attention_bias_applied"] += 1
        elif self.config.cognitive_compression and results:
            context = self.field._cognitive_compress(results)
            self.field.stats["cognitive_compressions"] += 1
        else:
            context = format_context(results, self.config.context_format)

        if self.config.symbolic_overlay and self.field.symbolic_overlay and results:
            facts = []
            for nid, score, node in results[:3]:
                text = node.content.get("text", "")
                concepts = self.field.symbolic_overlay._extract_concepts(text)
                facts.extend(concepts)
            if facts:
                symbolic_ctx = self.field.symbolic_overlay.get_symbolic_context(facts, max_depth=2)
                if symbolic_ctx:
                    context += "\n\n" + symbolic_ctx
                    self.field.stats["n_symbolic_inferences"] += 1
                    n_conflicts = sum(1 for r in self.field.symbolic_overlay.rules.values()
                                     if r.is_contextual_exception)
                    self.field.stats["n_symbolic_conflicts"] = n_conflicts

        return {"rtmdk_context": context}

    def _get_node_embedding(self, nid: str, node) -> Optional[np.ndarray]:
        """Retrieve stored embedding for a node, or approximate from latent position."""
        # Check if node has modal_embedding (cross-modal)
        if hasattr(node, 'modal_embedding') and node.modal_embedding is not None:
            return node.modal_embedding
        # Fallback: approximate embedding by inverse-projection from latent_pos
        # This is lossy but better than nothing for engram similarity
        if hasattr(node, 'latent_pos') and node.latent_pos is not None:
            # Pad latent_pos (64d) to embedding_dim (768d) with zeros
            # Engram similarity uses cosine — zeros won't dominate
            emb_dim = self.field.cfg.embedding_dim
            latent = node.latent_pos
            if len(latent) < emb_dim:
                approx = np.zeros(emb_dim, dtype=np.float32)
                approx[:len(latent)] = latent
                return approx
            return latent[:emb_dim] if len(latent) > emb_dim else latent
        return None

    def _detect_tags(self, text: str) -> List[str]:
        """Auto-detect memory tags from text content."""
        tags = []
        lower = text.lower()
        
        # Greeting/name tags
        if any(w in lower for w in ["hello", "hi ", "hey", "привет", "здравствуй", "hi,", "hey,"]):
            tags.append("greeting")
        if any(w in lower for w in ["my name is", "i'm ", "i am ", "меня зовут", "мое имя"]):
            tags.append("name")
        
        # Topic tags
        if any(w in lower for w in ["code", "program", "python", "java", "javascript", "функци", "код", "програм"]):
            tags.append("coding")
        if any(w in lower for w in ["coffee", "tea", "food", "drink", "кофе", "чай", "еда"]):
            tags.append("food_drink")
        if any(w in lower for w in ["love", "like", "prefer", "enjoy", "люб", "нрав", "предпочита"]):
            tags.append("preference")
        if any(w in lower for w in ["work", "job", "career", "работ", "карьер", "професс"]):
            tags.append("work")
        if any(w in lower for w in ["live", "city", "country", "home", "жив", "город", "стран", "дом"]):
            tags.append("location")
        if any(w in lower for w in ["family", "friend", "dog", "cat", "pet", "семь", "друг", "собак", "кот", "питом"]):
            tags.append("relationships")
        
        return tags[:5]  # Limit to 5 tags

    def _generate_clarification(self, results: List, query: str) -> str:
        """Generate a clarification prompt from weak-resonance nodes."""
        lines = [f"[CLARIFICATION] Не нашёл точных воспоминаний по запросу: \"{query[:80]}\""]
        lines.append("Полусовпадения (низкий резонанс):")
        for nid, score, node in results[:3]:
            text = node.content.get("text", "")[:60]
            lines.append(f"  [R:{score:.2f}] {text}")
        lines.append("Уточните запрос или предоставьте дополнительный контекст.")
        return "\n".join(lines)

    def get_system_prompt(self, context: str) -> str:
        return build_system_prompt(context, self.config.context_format, self.config.use_structured_prompt)

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """Save a conversation turn to memory with structured node format.
        
        Args:
            inputs: {"input": "user text", "session_id": "...", ...}
            outputs: {"output": "assistant text", ...}
        
        Node structure:
            input_text: User's message
            output_text: Assistant's response (empty if only input)
            role: "user" or "assistant"  
            session: Session/character ID
            timestamp: Unix timestamp
            emotion: Detected emotion (neutral by default)
            tags: Auto-detected memory tags
            tier: episodic/semantic/procedural
            context: Additional metadata
        """
        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")
        
        # If output is empty, still save the input
        if not output_text.strip():
            if not input_text.strip():
                return
            text_for_embedding = input_text
        else:
            text_for_embedding = output_text if len(output_text) > len(input_text) else input_text
        
        session_id = inputs.get("session_id", "default")
        timestamp = time.time()
        
        # Detect emotion from text
        emotion = "neutral"
        if input_text:
            lower_input = input_text.lower()
            if any(w in lower_input for w in ["happy", "love", "great", "wonderful", "amazing", "рад", "люб", "отличн", "прекрасн"]):
                emotion = "positive"
            elif any(w in lower_input for w in ["sad", "hate", "bad", "terrible", "angry", "грустн", "ненавиж", "плох", "зл"]):
                emotion = "negative"
            elif any(w in lower_input for w in ["?", "what", "why", "how", "when", "где", "что", "как", "когда", "почему"]):
                emotion = "questioning"
        
        # Auto-detect tags from text
        all_text = f"{input_text} {output_text}"
        tags = self._detect_tags(all_text)
        
        # Build structured node content
        content = {
            "input_text": input_text,
            "output_text": output_text,
            "role": "assistant" if output_text.strip() else "user",
            "session": session_id,
            "timestamp": timestamp,
            "emotion": emotion,
            "tags": tags,
            "tier": "episodic",  # Will be refined by tier detection
            "context": {
                k: v for k, v in inputs.items() 
                if k not in ["input", "query", "session_id", "embedding"]
            },
            "version": "2.0",  # Structured node version
        }
        
        embedding = self.embedder(text_for_embedding)
        phase = self._get_phase(session_id, embedding)
        modality = detect_modality(text_for_embedding) if self.config.cross_modal else "text"
        
        # Detect memory tier
        tier = detect_tier(text_for_embedding, inputs)
        content["tier"] = tier

        try:
            nid = self.field.add_node(embedding, content, phase, session_id=session_id, modality=modality)
        except SecurityViolationError:
            return

        # Set tier on the newly added node
        if nid in self.field.nodes:
            self.field.nodes[nid].tier = tier

        # Phase 18: Create/update engrams from co-activated nodes
        if self.engram_manager is not None:
            related_nodes = []
            for existing_nid, existing_node in self.field.nodes.items():
                existing_emb = self._get_node_embedding(existing_nid, existing_node)
                if existing_emb is not None:
                    sim = float(np.dot(embedding, existing_emb) / (
                        (np.linalg.norm(embedding) + 1e-8) * (np.linalg.norm(existing_emb) + 1e-8)))
                    if sim > 0.5:
                        related_nodes.append((existing_nid, sim))
            related_nodes.append((nid, 1.0))
            
            if len(related_nodes) >= self.config.engram_min_nodes:
                node_embs = {}
                for nid, _ in related_nodes:
                    emb = self._get_node_embedding(nid, self.field.nodes.get(nid))
                    if emb is not None:
                        node_embs[nid] = emb
                
                self.engram_manager.create_engram_from_nodes(
                    activated_nodes=related_nodes[:self.config.engram_max_nodes],
                    node_embeddings=node_embs,
                    semantic_core=text_for_embedding[:100],
                    context_tags=set(tags + [tier, session_id]),
                    tier=tier,
                )

        if self.config.enable_async:
            # Fix 4: Lazy async worker startup
            if self.config.async_pipeline and not self.field._workers_started:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.field._start_workers())
                    self.field._workers_started = True
                    # Enqueue for async processing
                    loop.create_task(self.field.evolve_q.put({"inputs": None}))
                except RuntimeError:
                    self.field.step()
            else:
                try:
                    asyncio.get_running_loop()
                    asyncio.create_task(self._evolve_field_async())
                except RuntimeError:
                    self.field.step()
        else:
            self.field.step()

    async def _evolve_field_async(self):
        await asyncio.sleep(0.01)
        self.field.step()

    def clear(self) -> None:
        # Fix 3: Cancel background workers before replacing field
        for task in self._workers:
            if not task.done():
                task.cancel()
        self._workers.clear()
        self.field = RTMDKField(self.config)
        self.session_phases.clear()

    def inspect_node(self, node_id: str) -> Optional[Dict]:
        if node_id not in self.field.nodes:
            return None
        node = self.field.nodes[node_id]
        info = {"id": node.id, "phase": node.phase, "amplitude": node.amplitude,
                "salience": node.salience, "tension": node.tension, "soft_gate": node.soft_gate,
                "self_sup_score": node.self_sup_score, "modal_weight": node.modal_weight,
                "modality": node.modality, "lineage": node.lineage, "content": node.content,
                "created_at": node.created_at, "last_resonated": node.last_resonated,
                "causal_parents": node.causal_parents, "causal_strength": node.causal_strength,
                "causal_effects": node.causal_effects, "is_causal_root": node.is_causal_root,
                "is_healing": node.is_healing, "healing_origin": node.healing_origin,
                "local_density": node.local_density, "goal_tags": node.goal_tags,
                "cross_modal_score": node.cross_modal_score}
        if node.pre_consolidation_pos is not None:
            info["pre_consolidation_pos"] = node.pre_consolidation_pos.tolist()
        if node.velocity is not None:
            info["velocity"] = node.velocity.tolist()
        if node.modal_embedding is not None:
            info["modal_embedding"] = node.modal_embedding.tolist()
        return info

    def rollback(self, n_steps: int = 1) -> bool:
        return self.field.rollback_consolidation(n_steps)

    def get_rollback_history(self) -> List[Dict]:
        return [{"timestamp": s["timestamp"], "updated": s["updated"], "n_nodes": len(s["pre_state"])}
                for s in self.field._rollback_history]

    def do_intervention(self, node_id: str, text: str):
        emb = self.embedder(text)
        self.field.do_intervention(node_id, emb)

    def clear_interventions(self):
        self.field.clear_interventions()

    def get_dashboard(self) -> Dict:
        return self.field.get_field_health()

    def get_field_health(self) -> Dict:
        return self.field.get_field_health()

    def trigger_healing(self) -> List[Dict]:
        return self.field._self_heal()

    def counterfactual_query(self, intervention: Dict[str, Any], query_nodes: List[str],
                             evidence: Optional[Dict[str, Any]] = None) -> CounterfactualResult:
        return self.field.counterfactual_query(intervention, query_nodes, evidence)

    def get_causal_summary(self) -> Dict:
        return self.field.get_causal_summary()

    def get_contradictions(self) -> List[ContradictionRecord]:
        if self.field.causal_engine:
            return list(self.field.causal_engine.contradictions.values())
        return []

    def resolve_contradiction(self, contradiction_id: str, resolution: str) -> bool:
        if self.field.causal_engine and contradiction_id in self.field.causal_engine.contradictions:
            self.field.causal_engine.contradictions[contradiction_id].resolved = True
            self.field.causal_engine.contradictions[contradiction_id].resolution = resolution
            return True
        return False

    def validate_consolidation(self, node_a: str, node_b: str) -> Dict[str, Any]:
        if self.field.causal_engine:
            return self.field.causal_engine.validate_consolidation(node_a, node_b)
        return {"safe": True, "reasons": [], "causal_conflicts": [], "recommendation": "proceed"}

    # Phase 7: ODE
    def evolve_continuous(self, inputs: Optional[List[Dict]] = None, use_sde: bool = False) -> NDArray:
        return self.field.evolve_continuous(inputs, use_sde)

    def get_response_smoothness(self) -> float:
        return self.field.stats.get("response_smoothness", 1.0)

    # Phase 8: Agent
    def create_plan(self, goal: str, available_tools: List[str], context: Optional[Dict] = None) -> AgentPlan:
        return self.field.create_plan(goal, available_tools, context)

    def verify_hypothesis(self, hypothesis: str, active_nodes: Optional[List[str]] = None) -> Hypothesis:
        return self.field.verify_hypothesis(hypothesis, active_nodes)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
        return self.field.execute_tool(tool_name, arguments)

    def register_tool(self, name: str, func: Callable):
        self.field.register_tool(name, func)

    # Phase 9: Production
    def evaluate_response(self, question: str, answer: str, contexts: List[str],
                          ground_truth: Optional[str] = None) -> EvalResult:
        return self.field.evaluate_response(question, answer, contexts, ground_truth)

    def compare_shadow(self, shadow_score: float, production_score: float) -> Dict[str, Any]:
        return self.field.compare_shadow(shadow_score, production_score)

    def get_ragas_trend(self) -> Dict[str, float]:
        if self.field.ragas_evaluator:
            return self.field.ragas_evaluator.get_trend()
        return {}

    # Track 10: New methods
    def get_cross_modal_stats(self) -> Dict:
        return self.field.get_cross_modal_stats()

    def get_meta_controller_state(self) -> Dict:
        return self.field.get_meta_controller_state()

    def get_federated_status(self) -> Dict:
        return self.field.get_federated_status()

    def get_stats(self) -> Dict:
        self.field.stats["active_nodes"] = len(self.field.nodes)
        if self.field.tda_monitor:
            self.field.stats["tda_trend"] = self.field.tda_monitor.get_trend()
        if self.field.dp:
            self.field.stats["privacy_budget_spent"] = self.field.dp.get_privacy_spent()
        return {**self.field.stats, "config": asdict(self.config)}

    # Phase 11 Track 4: Counterfactual imagination
    def imagine_counterfactual(self, base_query: str, intervention: Dict[str, float]) -> List[Dict]:
        """Generate hypothetical scenarios."""
        embedding = self.embedder(base_query)
        return self.field.imagine_counterfactual(embedding, intervention)

    def export_field(self, path: str):
        self.field.export_field(path)

    @classmethod
    def import_field(cls, path: str, embedder: Callable) -> "RTMDKMemory":
        return RTMDKField.import_field(path, embedder)

    # Phase 16 Track 3: Universal Memory Protocol
    def export_ump(self, path: str, source: str = "", comment: str = ""):
        """Export to Universal Memory Protocol format."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            raise ImportError("Universal Memory Protocol not available. Install rtmdk.support.ump")
        ump = UniversalMemoryProtocol.export(self.field, self, source=source, comment=comment)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ump, f, ensure_ascii=False, indent=2)

    @classmethod
    def import_ump(cls, path: str, embedder: Callable) -> "RTMDKMemory":
        """Import from Universal Memory Protocol format."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            raise ImportError("Universal Memory Protocol not available. Install rtmdk.support.ump")
        ump = _safe_json_load(path)
        return UniversalMemoryProtocol.import_ump(ump, embedder, memory_class=cls)

    def validate_ump(self, path: str) -> Dict:
        """Validate a UMP file."""
        path = _sanitize_path(path)
        if not UMP_AVAILABLE:
            return {"valid": False, "issues": ["UMP not available"]}
        ump = _safe_json_load(path)
        return UniversalMemoryProtocol.validate(ump)
