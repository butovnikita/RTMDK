"""
rtmdk/config.py
Configuration dataclass and enums for RTMDK.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
import logging

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
    latent_dim: int = 64
    resonance_kernel: str = "gaussian_phase"
    phase_coupling: float = 0.3
    bandwidth: float = 1.0
    attraction_lr: float = 0.02
    phase_sync_lr: float = 0.01
    decay_rate: float = 0.998
    min_amplitude: float = 0.05
    tension_threshold: float = 0.25
    consolidation_mode: ConsolidationMode = ConsolidationMode.DIALECTICAL
    max_nodes: Optional[int] = 5000
    top_k: int = 5
    min_response: float = 0.1
    enable_async: bool = True
    log_level: str = "INFO"

    # Phase 1
    context_format: ContextFormat = ContextFormat.PLAIN
    use_structured_prompt: bool = True
    adaptive_threshold: bool = False
    adaptive_window: int = 30
    learn_projection: bool = False
    projection_lr: float = 0.001
    projection_update_freq: int = 50
    pca_n_components: Optional[int] = None
    bm25_fallback: bool = False
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

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
    max_rollback_history: int = 50

    # Phase 3
    multimodal: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text"])
    modality_phase_shifts: Dict[str, float] = field(default_factory=dict)
    use_hnsw: bool = False
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
        "code": 3.14159 / 4,
        "audio": 3.14159 / 2,
        "vision": 3 * 3.14159 / 4,
        "metrics": 3.14159,
    })
    cross_modal_kernel_weight: float = 0.35

    # Track 10.2: Meta-cognitive controller
    meta_controller: bool = False
    meta_optimization_freq: int = 500
    meta_n_trials: int = 20
    meta_optimize_params: List[str] = field(default_factory=lambda: [
        "decay_rate", "tension_threshold", "phase_coupling", "bandwidth"
    ])

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

    def __post_init__(self):
        logger.setLevel(getattr(logging, self.log_level.upper()))
        if not self.modality_phase_shifts:
            self.modality_phase_shifts = {
                "text": 0.0, "audio": 3.14159 / 3,
                "image": 3.14159 / 2, "video": 3.14159,
            }
        if self.pca_n_components is None:
            self.pca_n_components = self.latent_dim
