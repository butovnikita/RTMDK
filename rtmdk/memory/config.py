"""
rtmdk/memory/config.py
RTMDK Configuration and Enums — extracted from core.py (P0 refactor).
"""

import os
import logging
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Callable, Any, Set, FrozenSet
import numpy as np

logger = logging.getLogger(__name__)


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

    # Phase 21: Self-Organizing Tokenizer + Embedding Field
    sot_enabled: bool = False
    sot_token_dim: Optional[int] = None  # None = same as latent_dim
    sot_max_vocab: int = 4096
    sot_merge_threshold: float = 0.7
    sot_contrastive_lr: float = 0.01
    sot_negatives_per_query: int = 5
    sot_ssm_sync: bool = False
    sot_diagonal_ssm: bool = True  # O(N*d) instead of O(N*d^2)
    sot_merge_freq: int = 100
    sot_min_cooccurrence: int = 5
    sot_use_for_query: bool = False

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
