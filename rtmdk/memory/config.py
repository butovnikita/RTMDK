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


# ──── Field-to-group mapping for flat-field access ────
_GROUP_ATTRS: Dict[str, str] = {
    "CoreConfig": "core",
    "RetrievalConfig": "retrieval",
    "LearningConfig": "learning",
    "DynamicsConfig": "dynamics",
    "InferenceConfig": "inference",
    "MemorySystemConfig": "memory",
    "ProductionConfig": "production",
    "RoutingConfig": "routing",
    "SOTConfig": "sot",
}

_FIELD_GROUPS: Dict[str, str] = {
    "adaptive_threshold": "CoreConfig",
    "adaptive_window": "CoreConfig",
    "adjoint_enabled": "DynamicsConfig",
    "agent_orchestration": "InferenceConfig",
    "async_pipeline": "MemorySystemConfig",
    "attention_bias": "RetrievalConfig",
    "attention_tokens": "MemorySystemConfig",
    "attraction_lr": "CoreConfig",
    "auto_role_detection": "MemorySystemConfig",
    "auto_rollback": "ProductionConfig",
    "auto_rollback_threshold": "ProductionConfig",
    "backend": "RetrievalConfig",
    "ball_radius": "MemorySystemConfig",
    "adaptive_bandwidth": "CoreConfig",
    "adaptive_bandwidth_k": "CoreConfig",
    "conformal_prediction": "CoreConfig",
    "conformal_alpha": "CoreConfig",
    "conformal_min_calib": "CoreConfig",
    "spectral_consolidation": "CoreConfig",
    "spectral_max_clusters": "CoreConfig",
    "spectral_sigma": "CoreConfig",
    "enable_kalman_filter": "CoreConfig",
    "kalman_diagonal_approx": "CoreConfig",
    "kalman_process_noise": "CoreConfig",
    "kalman_measurement_noise": "CoreConfig",
    "kalman_init_variance": "CoreConfig",
    "bandwidth": "CoreConfig",
    "bias_temperature": "RetrievalConfig",
    "bm25_b": "CoreConfig",
    "bm25_fallback": "CoreConfig",
    "bm25_k1": "CoreConfig",
    "causal_adjustment_sets": "InferenceConfig",
    "causal_discovery_min_samples": "InferenceConfig",
    "causal_graph_integrity_check": "ProductionConfig",
    "causal_masking": "InferenceConfig",
    "causal_max_hops": "DynamicsConfig",
    "causal_p_threshold": "InferenceConfig",
    "causal_topological": "InferenceConfig",
    "causal_traversal": "DynamicsConfig",
    "clarification_threshold_ratio": "MemorySystemConfig",
    "cognitive_compression": "MemorySystemConfig",
    "compression_freq": "InferenceConfig",
    "compression_rank": "InferenceConfig",
    "consolidation_loss_weight": "LearningConfig",
    "consolidation_mode": "CoreConfig",
    "context_format": "CoreConfig",
    "continuous_dynamics": "DynamicsConfig",
    "contradiction_detection": "InferenceConfig",
    "contradiction_threshold": "InferenceConfig",
    "counterfactual_enabled": "InferenceConfig",
    "counterfactual_imagination": "DynamicsConfig",
    "counterfactual_max_depth": "InferenceConfig",
    "cpen_child_ode": "MemorySystemConfig",
    "cpen_parent_ode": "MemorySystemConfig",
    "cross_modal": "RoutingConfig",
    "cross_modal_kernel_weight": "RoutingConfig",
    "cross_shard_threshold": "MemorySystemConfig",
    "crystallization": "MemorySystemConfig",
    "crystallization_freq": "MemorySystemConfig",
    "crystallization_min_cluster": "MemorySystemConfig",
    "crystallization_similarity": "MemorySystemConfig",
    "curvature": "MemorySystemConfig",
    "dead_zone_threshold": "LearningConfig",
    "decay_rate": "CoreConfig",
    "differentiable": "LearningConfig",
    "differential_privacy": "ProductionConfig",
    "do_calculus_validation": "InferenceConfig",
    "domain_aware_retrieval": "MemorySystemConfig",
    "domain_consolidation_guard": "MemorySystemConfig",
    "dp_delta": "ProductionConfig",
    "dp_epsilon": "ProductionConfig",
    "dp_max_norm": "ProductionConfig",
    "dreaming_freq": "DynamicsConfig",
    "drift_detection": "ProductionConfig",
    "drift_threshold": "ProductionConfig",
    "drift_window": "ProductionConfig",
    "embedding_dim": "CoreConfig",
    "enable_async": "CoreConfig",
    "enable_engrams": "MemorySystemConfig",
    "enable_rollback": "RetrievalConfig",
    "engram_creation_threshold": "MemorySystemConfig",
    "engram_decay_rate": "MemorySystemConfig",
    "engram_max_nodes": "MemorySystemConfig",
    "engram_min_nodes": "MemorySystemConfig",
    "engram_overlap_threshold": "MemorySystemConfig",
    "engram_pattern_completion": "MemorySystemConfig",
    "entropy_high_threshold": "MemorySystemConfig",
    "entropy_low_threshold": "MemorySystemConfig",
    "entropy_management": "MemorySystemConfig",
    "eval_frequency": "ProductionConfig",
    "eval_mode": "ProductionConfig",
    "event_driven": "InferenceConfig",
    "evolve_queue_size": "MemorySystemConfig",
    "false_merge_threshold": "RetrievalConfig",
    "federated": "RoutingConfig",
    "federated_min_resonance": "RoutingConfig",
    "federated_sync_freq": "RoutingConfig",
    "federated_sync_lr": "RoutingConfig",
    "field_stability_window": "RetrievalConfig",
    "fragmentation_threshold": "LearningConfig",
    "gate_temperature": "RetrievalConfig",
    "goal_completion_threshold": "InferenceConfig",
    "goal_decay": "InferenceConfig",
    "goal_directed_routing": "InferenceConfig",
    "goal_tracking": "InferenceConfig",
    "gpu_batch_size": "RetrievalConfig",
    "gradient_clip": "LearningConfig",
    "healing_check_freq": "LearningConfig",
    "healing_strength": "LearningConfig",
    "hebbian_learning_rate": "MemorySystemConfig",
    "high_resonance_threshold": "MemorySystemConfig",
    "hnsw_ef_construction": "RetrievalConfig",
    "hnsw_m": "RetrievalConfig",
    "hybrid_alpha": "CoreConfig",
    "hyperbolic": "MemorySystemConfig",
    "hyperconvergence_threshold": "LearningConfig",
    "hypothesis_verification": "InferenceConfig",
    "kurtosis_target_max": "LearningConfig",
    "kurtosis_target_min": "LearningConfig",
    "l2_regularization": "RetrievalConfig",
    "latent_dim": "CoreConfig",
    "learn_projection": "CoreConfig",
    "learnable_bandwidth": "LearningConfig",
    "learnable_decay": "LearningConfig",
    "learnable_phase_coupling": "LearningConfig",
    "log_level": "CoreConfig",
    "low_rank_compression": "InferenceConfig",
    "lyapunov_alpha": "MemorySystemConfig",
    "lyapunov_beta": "MemorySystemConfig",
    "lyapunov_gamma": "MemorySystemConfig",
    "lyapunov_threshold": "MemorySystemConfig",
    "max_goals": "InferenceConfig",
    "max_healing_nodes_per_step": "LearningConfig",
    "max_node_text_length": "ProductionConfig",
    "max_nodes": "CoreConfig",
    "max_plan_depth": "InferenceConfig",
    "max_rollback_history": "RetrievalConfig",
    "max_scenarios": "DynamicsConfig",
    "max_tool_calls": "InferenceConfig",
    "max_versions": "MemorySystemConfig",
    "memory_age_factor": "MemorySystemConfig",
    "memory_tiers": "MemorySystemConfig",
    "meta_adaptation_lr": "LearningConfig",
    "meta_adaptive": "LearningConfig",
    "meta_controller": "InferenceConfig",
    "meta_memory": "MemorySystemConfig",
    "meta_n_trials": "InferenceConfig",
    "meta_optimization_freq": "InferenceConfig",
    "meta_optimize_params": "InferenceConfig",
    "metrics_retention": "ProductionConfig",
    "min_amplitude": "CoreConfig",
    "min_nodes_for_gpu": "RoutingConfig",
    "min_response": "CoreConfig",
    "modal_phase_offsets": "RoutingConfig",
    "modalities": "RetrievalConfig",
    "modality_phase_shifts": "RetrievalConfig",
    "multimodal": "RetrievalConfig",
    "neuro_symbolic_prover": "InferenceConfig",
    "node_id": "RoutingConfig",
    "num_shards": "RoutingConfig",
    "ode_atol": "DynamicsConfig",
    "ode_chunk_size": "DynamicsConfig",
    "ode_n_steps": "DynamicsConfig",
    "ode_rtol": "DynamicsConfig",
    "ode_solver": "DynamicsConfig",
    "ode_time_horizon": "DynamicsConfig",
    "offline_dreaming": "DynamicsConfig",
    "pc_latent_dim": "DynamicsConfig",
    "pc_lr": "DynamicsConfig",
    "pca_n_components": "CoreConfig",
    "phase_coupling": "CoreConfig",
    "phase_sync_lr": "CoreConfig",
    "predictive_coding": "DynamicsConfig",
    "proactive_clarification": "MemorySystemConfig",
    "production_mode": "ProductionConfig",
    "projection_lr": "CoreConfig",
    "projection_update_freq": "CoreConfig",
    "prompt_injection_patterns": "ProductionConfig",
    "prover_backend": "InferenceConfig",
    "query_queue_size": "MemorySystemConfig",
    "ragas_enabled": "ProductionConfig",
    "recall_accuracy_threshold": "MemorySystemConfig",
    "resonance_kernel": "CoreConfig",
    "response_smoothness_target": "DynamicsConfig",
    "rl_feedback": "InferenceConfig",
    "rl_learning_rate": "InferenceConfig",
    "rl_reward_window": "InferenceConfig",
    "role_sharding": "MemorySystemConfig",
    "role_shards": "MemorySystemConfig",
    "safety_certifier": "MemorySystemConfig",
    "safety_mode": "MemorySystemConfig",
    "save_queue_size": "MemorySystemConfig",
    "sde_noise_level": "DynamicsConfig",
    "security_enabled": "ProductionConfig",
    "seed": "CoreConfig",
    "self_healing": "LearningConfig",
    "self_reflection_freq": "MemorySystemConfig",
    "self_sup_threshold": "RetrievalConfig",
    "self_sup_verify_after_consolidate": "RetrievalConfig",
    "self_supervision": "RetrievalConfig",
    "shadow_fallback_threshold": "ProductionConfig",
    "shadow_mode": "ProductionConfig",
    "soft_gates": "RetrievalConfig",
    "sot_contrastive_lr": "SOTConfig",
    "sot_diagonal_ssm": "SOTConfig",
    "sot_enabled": "SOTConfig",
    "sot_max_vocab": "SOTConfig",
    "sot_merge_freq": "SOTConfig",
    "sot_merge_threshold": "SOTConfig",
    "sot_min_cooccurrence": "SOTConfig",
    "sot_negatives_per_query": "SOTConfig",
    "sot_ssm_sync": "SOTConfig",
    "sot_token_dim": "SOTConfig",
    "sot_use_for_query": "SOTConfig",
    "sot_warm_start_corpus": "SOTConfig",
    "sot_attention_pooling": "SOTConfig",
    "sot_hard_negatives": "SOTConfig",
    "sot_retrieval_feedback": "SOTConfig",
    "sot_skipgram_window": "SOTConfig",
    "sot_subword_seed": "SOTConfig",
    "sot_bootstrap_projection": "SOTConfig",
    "sot_bootstrap_corpus": "SOTConfig",
    "sot_bootstrap_model": "SOTConfig",
    "sot_tokenization_mode": "SOTConfig",
    "sot_max_cooccurrence": "SOTConfig",
    "sparse_routing": "RoutingConfig",
    "ssm_dynamics": "DynamicsConfig",
    "ssm_state_dim": "DynamicsConfig",
    "swarm_consensus_threshold": "RoutingConfig",
    "swarm_max_agents": "RoutingConfig",
    "swarm_memory": "RoutingConfig",
    "swarm_vote_weight": "RoutingConfig",
    "symbolic_confidence_threshold": "MemorySystemConfig",
    "symbolic_max_tension": "MemorySystemConfig",
    "symbolic_min_self_sup": "MemorySystemConfig",
    "symbolic_overlay": "MemorySystemConfig",
    "system_prompt": "ProductionConfig",
    "tda_check_freq": "RetrievalConfig",
    "tda_monitoring": "RetrievalConfig",
    "tension_spike_threshold": "ProductionConfig",
    "tension_threshold": "CoreConfig",
    "tier_decay": "MemorySystemConfig",
    "tier_tension_thresh": "MemorySystemConfig",
    "tool_timeout": "InferenceConfig",
    "top_k": "CoreConfig",
    "top_shards": "RoutingConfig",
    "triton_backend": "RoutingConfig",
    "trust_consensus": "InferenceConfig",
    "trust_min_reputation": "InferenceConfig",
    "ump_enabled": "MemorySystemConfig",
    "use_hnsw": "RetrievalConfig",
    "use_structured_prompt": "CoreConfig",
    "verification_confidence_threshold": "InferenceConfig",
    "version_control": "MemorySystemConfig",
}

@dataclass
class CoreConfig:
        embedding_dim: int = 768
        latent_dim: int = 64  # Matches server default — change only if you know the impact
        resonance_kernel: str = "gaussian_phase"
        phase_coupling: float = 0.3
        bandwidth: float = 1.0
        adaptive_bandwidth: bool = False
        adaptive_bandwidth_k: int = 5
        conformal_prediction: bool = False
        conformal_alpha: float = 0.10
        conformal_min_calib: int = 50
        spectral_consolidation: bool = False
        spectral_max_clusters: int = 10
        spectral_sigma: float = 1.0
        enable_kalman_filter: bool = False
        kalman_diagonal_approx: bool = True
        kalman_process_noise: float = 0.01
        kalman_measurement_noise: float = 0.1
        kalman_init_variance: float = 1.0
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


@dataclass
class RetrievalConfig:
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
        multimodal: bool = False
        modalities: List[str] = field(default_factory=lambda: ["text"])
        modality_phase_shifts: Dict[str, float] = field(default_factory=dict)
        use_hnsw: bool = True  # OPTIMIZED: fast approximate nearest neighbor
        hnsw_m: int = 16
        hnsw_ef_construction: int = 200
        tda_monitoring: bool = False
        tda_check_freq: int = 50
        attention_bias: bool = False
        bias_temperature: float = 1.0


@dataclass
class LearningConfig:
        differentiable: bool = False
        learnable_bandwidth: bool = False
        learnable_phase_coupling: bool = False
        learnable_decay: bool = False
        gradient_clip: float = 1.0
        consolidation_loss_weight: float = 0.1
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


@dataclass
class DynamicsConfig:
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
        predictive_coding: bool = False
        pc_latent_dim: int = 32
        pc_lr: float = 0.01
        counterfactual_imagination: bool = False
        max_scenarios: int = 5
        offline_dreaming: bool = True
        dreaming_freq: int = 50
        causal_traversal: bool = True
        causal_max_hops: int = 3
        ssm_dynamics: bool = False
        ssm_state_dim: int = 64


@dataclass
class InferenceConfig:
        causal_topological: bool = False
        causal_discovery_min_samples: int = 20
        causal_p_threshold: float = 0.05
        do_calculus_validation: bool = True
        counterfactual_enabled: bool = False
        counterfactual_max_depth: int = 3
        contradiction_detection: bool = True
        contradiction_threshold: float = 0.3
        causal_adjustment_sets: bool = True
        agent_orchestration: bool = False
        max_plan_depth: int = 3
        max_tool_calls: int = 5
        tool_timeout: float = 15.0
        hypothesis_verification: bool = True
        verification_confidence_threshold: float = 0.7
        goal_directed_routing: bool = False
        meta_controller: bool = False
        meta_optimization_freq: int = 500
        meta_n_trials: int = 20
        meta_optimize_params: List[str] = field(default_factory=lambda: [
            "decay_rate", "tension_threshold", "phase_coupling", "bandwidth"
        ])
        goal_tracking: bool = False
        max_goals: int = 20
        goal_decay: float = 0.995
        goal_completion_threshold: float = 0.8
        rl_feedback: bool = False
        rl_learning_rate: float = 0.01
        rl_reward_window: int = 10
        event_driven: bool = False
        low_rank_compression: bool = False
        compression_rank: int = 32
        compression_freq: int = 500
        trust_consensus: bool = False
        trust_min_reputation: float = 0.3
        neuro_symbolic_prover: bool = False
        prover_backend: str = "z3"
        causal_masking: bool = False


@dataclass
class MemorySystemConfig:
        enable_engrams: bool = True
        engram_min_nodes: int = 2
        engram_max_nodes: int = 20
        engram_creation_threshold: float = 0.6
        engram_decay_rate: float = 0.998
        engram_pattern_completion: bool = True
        engram_overlap_threshold: float = 0.7
        memory_tiers: Set[str] = field(default_factory=lambda: {"episodic", "semantic", "procedural"})
        tier_decay: Dict[str, float] = field(default_factory=lambda: {
            "episodic": 0.992, "semantic": 0.999, "procedural": 1.0
        })
        tier_tension_thresh: Dict[str, float] = field(default_factory=lambda: {
            "episodic": 0.10, "semantic": 0.22, "procedural": 0.35
        })
        hyperbolic: bool = False
        ball_radius: float = 0.85
        curvature: float = -1.0
        cognitive_compression: bool = False
        high_resonance_threshold: float = 0.6
        crystallization: bool = False
        crystallization_freq: int = 200
        crystallization_similarity: float = 0.75
        crystallization_min_cluster: int = 3
        async_pipeline: bool = False
        query_queue_size: int = 50
        save_queue_size: int = 100
        evolve_queue_size: int = 20
        meta_memory: bool = False
        self_reflection_freq: int = 100
        memory_age_factor: float = 0.001
        recall_accuracy_threshold: float = 0.6
        version_control: bool = False
        max_versions: int = 100
        proactive_clarification: bool = False
        clarification_threshold_ratio: float = 0.5
        attention_tokens: bool = True  # Enabled by default (extends attention_bias)
        entropy_management: bool = False
        entropy_high_threshold: float = 3.0
        entropy_low_threshold: float = 0.5
        symbolic_overlay: bool = False
        symbolic_min_self_sup: float = 0.7
        symbolic_max_tension: float = 0.15
        symbolic_confidence_threshold: float = 0.65
        safety_certifier: bool = False
        safety_mode: str = "soft_regulate"  # monitor_only | soft_regulate | hard_block
        lyapunov_alpha: float = 0.4
        lyapunov_beta: float = 0.4
        lyapunov_gamma: float = 0.2
        lyapunov_threshold: float = 0.1
        ump_enabled: bool = False
        role_sharding: bool = False
        role_shards: Set[str] = field(default_factory=lambda: {"default"})
        cross_shard_threshold: float = 0.45
        auto_role_detection: bool = True
        domain_aware_retrieval: bool = False
        domain_consolidation_guard: bool = False
        cpen_parent_ode: bool = False
        cpen_child_ode: bool = False
        hebbian_learning_rate: float = 0.01


@dataclass
class ProductionConfig:
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
        differential_privacy: bool = False
        dp_epsilon: float = 2.0
        dp_delta: float = 1e-5
        dp_max_norm: float = 1.0
        security_enabled: bool = False
        max_node_text_length: int = 10000
        tension_spike_threshold: float = 0.5
        causal_graph_integrity_check: bool = True
        prompt_injection_patterns: List[str] = field(default_factory=lambda: [
            "ignore previous", "system prompt", "you are now", "disregard",
            "ignore all", "new instruction", "override"
        ])
        system_prompt: Optional[str] = "You are a helpful assistant with long-term memory powered by RTMDK (Resonance-Topological Memory)."


@dataclass
class RoutingConfig:
        cross_modal: bool = False
        modal_phase_offsets: Dict[str, float] = field(default_factory=lambda: {
            "text": 0.0,
            "code": math.pi / 4,
            "audio": math.pi / 2,
            "vision": 3 * math.pi / 4,
            "metrics": math.pi,
        })
        cross_modal_kernel_weight: float = 0.35
        federated: bool = False
        federated_sync_lr: float = 0.01
        federated_sync_freq: int = 100
        federated_min_resonance: float = 0.2
        node_id: str = "local"
        sparse_routing: bool = False
        num_shards: int = 8
        top_shards: int = 3
        swarm_memory: bool = False
        swarm_consensus_threshold: float = 0.5
        swarm_max_agents: int = 10
        swarm_vote_weight: float = 0.3
        triton_backend: bool = False
        min_nodes_for_gpu: int = 2000


@dataclass
class SOTConfig:
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
        sot_warm_start_corpus: Optional[str] = None  # Path to JSON corpus for warm-start
        sot_attention_pooling: bool = False  # IDF + position weighted pooling
        sot_hard_negatives: bool = False  # Use hardest negatives in contrastive learning
        sot_retrieval_feedback: bool = False  # Update embeddings from query feedback
        sot_skipgram_window: int = 1  # 1=adjacent only, >1=skip-gram window
        sot_subword_seed: bool = False  # Pre-seed with common byte n-grams
        sot_bootstrap_projection: Optional[str] = None  # Path to .npz bootstrap file
        sot_bootstrap_corpus: Optional[str] = None  # Path to JSON corpus for auto-bootstrap
        sot_bootstrap_model: str = "all-MiniLM-L6-v2"  # Teacher model for auto-bootstrap
        sot_tokenization_mode: str = "byte"  # "byte" or "word"
        sot_max_cooccurrence: int = 100_000  # Max co-occurrence entries before pruning


@dataclass(init=False, repr=False, eq=False)
class RTMDKConfig:
    core: CoreConfig = field(default_factory=CoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    memory: MemorySystemConfig = field(default_factory=MemorySystemConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    sot: SOTConfig = field(default_factory=SOTConfig)

    def __init__(self, **kwargs):
        """Backward-compatible constructor accepting both flat and nested kwargs."""
        # Initialize nested configs with defaults
        self.core = CoreConfig()
        self.retrieval = RetrievalConfig()
        self.learning = LearningConfig()
        self.dynamics = DynamicsConfig()
        self.inference = InferenceConfig()
        self.memory = MemorySystemConfig()
        self.production = ProductionConfig()
        self.routing = RoutingConfig()
        self.sot = SOTConfig()

        # Handle nested config objects passed directly
        nested_keys = {k for k in kwargs if not _FIELD_GROUPS.get(k)}
        for key in list(kwargs.keys()):
            if key in nested_keys and hasattr(self, key):
                setattr(self, key, kwargs.pop(key))

        # Apply flat-field overrides
        for key, value in kwargs.items():
            self._set_flat_field(key, value)

        self.__post_init__()

    def _set_flat_field(self, name: str, value: Any) -> None:
        group = _FIELD_GROUPS.get(name)
        if group is None:
            raise AttributeError(f"Unknown config field: {name}")
        nested = getattr(self, _GROUP_ATTRS[group])
        object.__setattr__(nested, name, value)

    def _get_flat_field(self, name: str) -> Any:
        group = _FIELD_GROUPS.get(name)
        if group is None:
            raise AttributeError(f"Unknown config field: {name}")
        nested = getattr(self, _GROUP_ATTRS[group])
        return getattr(nested, name)

    def __getattr__(self, name: str) -> Any:
        if name in _FIELD_GROUPS:
            return self._get_flat_field(name)
        raise AttributeError(f"RTMDKConfig has no attribute {name!r}")

    _NESTED_ATTRS = frozenset(
        {"core", "retrieval", "learning", "dynamics", "inference",
         "memory", "production", "routing", "sot"}
    )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _FIELD_GROUPS:
            self._set_flat_field(name, value)
        elif name in self._NESTED_ATTRS:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"RTMDKConfig has no attribute {name!r}")

    def __repr__(self) -> str:
        parts = []
        parts.append(f"core={self.core!r}")
        parts.append(f"retrieval={self.retrieval!r}")
        parts.append(f"learning={self.learning!r}")
        parts.append(f"dynamics={self.dynamics!r}")
        parts.append(f"inference={self.inference!r}")
        parts.append(f"memory={self.memory!r}")
        parts.append(f"production={self.production!r}")
        parts.append(f"routing={self.routing!r}")
        parts.append(f"sot={self.sot!r}")
        return "RTMDKConfig(" + ", ".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RTMDKConfig):
            return NotImplemented
        if self.core != other.core:
            return False
        if self.retrieval != other.retrieval:
            return False
        if self.learning != other.learning:
            return False
        if self.dynamics != other.dynamics:
            return False
        if self.inference != other.inference:
            return False
        if self.memory != other.memory:
            return False
        if self.production != other.production:
            return False
        if self.routing != other.routing:
            return False
        if self.sot != other.sot:
            return False
        return True

    def asdict(self) -> Dict[str, Any]:
        """Flat dictionary of all config fields (backward compatible)."""
        result: Dict[str, Any] = {}
        result.update(vars(self.core))
        result.update(vars(self.retrieval))
        result.update(vars(self.learning))
        result.update(vars(self.dynamics))
        result.update(vars(self.inference))
        result.update(vars(self.memory))
        result.update(vars(self.production))
        result.update(vars(self.routing))
        result.update(vars(self.sot))
        return result

    def __post_init__(self):
        """Env var overrides and derived defaults."""

        _env_overrides = [
            ("RTMDK_EMBEDDING_DIM", "embedding_dim", int),
            ("RTMDK_LATENT_DIM", "latent_dim", int),
            ("RTMDK_DECAY_RATE", "decay_rate", float),
            ("RTMDK_TENSION_THRESHOLD", "tension_threshold", float),
            ("RTMDK_MIN_RESPONSE", "min_response", float),
            ("RTMDK_TOP_K", "top_k", int),
            ("RTMDK_MAX_NODES", "max_nodes", lambda x: int(x) if x and x.lower() != "none" else None),
            ("RTMDK_CONSOLIDATION_MODE", "consolidation_mode", lambda x: ConsolidationMode(x)),
            ("RTMDK_PHASE_COUPLING", "phase_coupling", float),
            ("RTMDK_BANDWIDTH", "bandwidth", float),
            ("RTMDK_ADAPTIVE_BANDWIDTH", "adaptive_bandwidth", lambda x: x.lower() == "true"),
            ("RTMDK_ADAPTIVE_BANDWIDTH_K", "adaptive_bandwidth_k", int),
            ("RTMDK_CONFORMAL_PREDICTION", "conformal_prediction", lambda x: x.lower() == "true"),
            ("RTMDK_CONFORMAL_ALPHA", "conformal_alpha", float),
            ("RTMDK_CONFORMAL_MIN_CALIB", "conformal_min_calib", int),
            ("RTMDK_SPECTRAL_CONSOLIDATION", "spectral_consolidation", lambda x: x.lower() == "true"),
            ("RTMDK_SPECTRAL_MAX_CLUSTERS", "spectral_max_clusters", int),
            ("RTMDK_SPECTRAL_SIGMA", "spectral_sigma", float),
            ("RTMDK_ENABLE_KALMAN_FILTER", "enable_kalman_filter", lambda x: x.lower() == "true"),
            ("RTMDK_KALMAN_DIAGONAL_APPROX", "kalman_diagonal_approx", lambda x: x.lower() == "true"),
            ("RTMDK_KALMAN_PROCESS_NOISE", "kalman_process_noise", float),
            ("RTMDK_KALMAN_MEASUREMENT_NOISE", "kalman_measurement_noise", float),
            ("RTMDK_KALMAN_INIT_VARIANCE", "kalman_init_variance", float),
            ("RTMDK_USE_HNSW", "use_hnsw", lambda x: x.lower() == "true"),
            ("RTMDK_HNSW_M", "hnsw_m", int),
            ("RTMDK_BM25_FALLBACK", "bm25_fallback", lambda x: x.lower() == "true"),
            ("RTMDK_LEARN_PROJECTION", "learn_projection", lambda x: x.lower() == "true"),
            ("RTMDK_PROJECTION_LR", "projection_lr", float),
            ("RTMDK_PROJECTION_UPDATE_FREQ", "projection_update_freq", int),
            ("RTMDK_ATTENTION_BIAS", "attention_bias", lambda x: x.lower() == "true"),
            ("RTMDK_ENABLE_ASYNC", "enable_async", lambda x: x.lower() == "true"),
            ("RTMDK_SOFT_GATES", "soft_gates", lambda x: x.lower() == "true"),
            ("RTMDK_ADAPTIVE_THRESHOLD", "adaptive_threshold", lambda x: x.lower() == "true"),
            ("RTMDK_CROSS_MODAL", "cross_modal", lambda x: x.lower() == "true"),
            ("RTMDK_CAUSAL_TOPOLOGICAL", "causal_topological", lambda x: x.lower() == "true"),
            ("RTMDK_META_ADAPTIVE", "meta_adaptive", lambda x: x.lower() == "true"),
            ("RTMDK_SELF_HEALING", "self_healing", lambda x: x.lower() == "true"),
            ("RTMDK_VERSION_CONTROL", "version_control", lambda x: x.lower() == "true"),
            ("RTMDK_ENABLE_ENGRAMS", "enable_engrams", lambda x: x.lower() == "true"),
            ("RTMDK_ENGRAM_MIN_NODES", "engram_min_nodes", int),
            ("RTMDK_ENGRAM_MAX_NODES", "engram_max_nodes", int),
            ("RTMDK_OFFLINE_DREAMING", "offline_dreaming", lambda x: x.lower() == "true"),
            ("RTMDK_CAUSAL_TRAVERSAL", "causal_traversal", lambda x: x.lower() == "true"),
            ("RTMDK_CAUSAL_MAX_HOPS", "causal_max_hops", int),
            ("RTMDK_SSM_DYNAMICS", "ssm_dynamics", lambda x: x.lower() == "true"),
            ("RTMDK_SSM_STATE_DIM", "ssm_state_dim", int),
            ("RTMDK_TRUST_CONSENSUS", "trust_consensus", lambda x: x.lower() == "true"),
            ("RTMDK_NEURO_SYMBOLIC_PROVER", "neuro_symbolic_prover", lambda x: x.lower() == "true"),
            ("RTMDK_HYPERBOLIC", "hyperbolic", lambda x: x.lower() == "true"),
            ("RTMDK_PREDICTIVE_CODING", "predictive_coding", lambda x: x.lower() == "true"),
            ("RTMDK_COUNTERFACTUAL_IMAGINATION", "counterfactual_imagination", lambda x: x.lower() == "true"),
            ("RTMDK_DIFFERENTIAL_PRIVACY", "differential_privacy", lambda x: x.lower() == "true"),
            ("RTMDK_DP_EPSILON", "dp_epsilon", float),
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
            ("RTMDK_DOMAIN_AWARE_RETRIEVAL", "domain_aware_retrieval", lambda x: x.lower() == "true"),
            ("RTMDK_DOMAIN_CONSOLIDATION_GUARD", "domain_consolidation_guard", lambda x: x.lower() == "true"),
        ]

        for env_key, attr, type_fn in _env_overrides:
            val = os.getenv(env_key)
            if val is not None:
                try:
                    parsed = type_fn(val)
                    if attr == "system_prompt" and parsed == "":
                        parsed = None
                    elif attr == "system_prompt" and str(parsed).lower() == "none":
                        parsed = None
                    self._set_flat_field(attr, parsed)
                except (ValueError, TypeError) as e:
                    logging.getLogger("rtmdk").warning(
                        f"Invalid env var {env_key}={val}: {e}"
                    )

        logger.setLevel(getattr(logging, self.core.log_level.upper()))
        if not self.retrieval.modality_phase_shifts:
            self.retrieval.modality_phase_shifts = {
                "text": 0.0, "audio": np.pi / 3,
                "image": np.pi / 2, "video": np.pi,
            }
        if self.core.pca_n_components is None:
            self.core.pca_n_components = self.core.latent_dim
