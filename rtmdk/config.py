"""
rtmdk/config.py — Central Configuration for RTMDK.

ALL configuration variables in one place, organized by category.
Each variable includes a comment explaining its effect.

Categories:
  CORE — Fundamental RTMDK parameters
  RETRIEVAL — How memories are retrieved  
  PERFORMANCE — Speed and efficiency settings
  PRODUCTION — Production deployment settings
  SCALING — Settings for large-scale deployments (>100K nodes)

Presets:
  local()     — Single user, minimal resources
  production() — Multi-user, all optimizations
  research()   — Maximum accuracy, slower
  enterprise() — Distributed, 100K+ nodes
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
import logging


# ============================================================================
# ENUMS
# ============================================================================

class ConsolidationMode(Enum):
    DIALECTICAL = "dialectical"  # Best for quality — merges nodes intelligently
    MERGE = "merge"              # Faster — simple averaging
    PRUNE = "prune"              # Aggressive — removes weak nodes

class Backend(Enum):
    NUMPY = "numpy"   # CPU-only, works everywhere
    TORCH = "torch"   # GPU acceleration (needs CUDA)

class ContextFormat(Enum):
    PLAIN = "plain"    # Simple text — fastest for LLM
    JSON = "json"      # Structured — better for parsing
    YAML = "yaml"      # Human-readable — good for debugging
    ATTENTION = "attention"  # Control tokens — for attention-aware LLMs

class FieldHealth(Enum):
    STABLE = "stable"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    HEALING = "healing"

class EvalMode(Enum):
    PRODUCTION = "production"
    SHADOW = "shadow"
    EVALUATION = "evaluation"


# ============================================================================
# CENTRAL CONFIGURATION
# ============================================================================

@dataclass
class RTMDKConfig:
    """
    Complete RTMDK Configuration.
    
    All variables are documented with their effect.
    Use presets (local/production/research/enterprise) for common setups,
    or customize individual values.
    """

    # ─────────────────────────────────────────────────────────────────────
    # CORE — Fundamental parameters
    # ─────────────────────────────────────────────────────────────────────

    embedding_dim: int = 768
    """Size of input embeddings. Must match your embedder model.
    Nomic embed-text-v1.5 = 768, all-MiniLM-L6-v2 = 384.
    ↓ Smaller = faster, ↑ Larger = more accurate."""

    latent_dim: int = 256
    """Internal representation size. Larger = more detail retained.
    Default 256: good balance. 64: faster but lossy. 512: max accuracy.
    ↑ Larger = better recall but more RAM and slower."""

    decay_rate: float = 0.999
    """How fast memories fade. 0.999 = very slow (half-life ~693 steps).
    0.995 = moderate forgetting (half-life ~138 steps).
    0.990 = aggressive forgetting (half-life ~69 steps).
    ↓ Lower = faster forgetting, ↑ Higher = longer retention."""

    min_response: float = 0.005
    """Minimum resonance score to include in results.
    0.005 = include weak matches, 0.01 = moderate, 0.1 = strict.
    ↓ Lower = more results (including noise), ↑ Higher = fewer but stronger."""

    top_k: int = 5
    """Number of memory nodes to return per query.
    3 = concise context, 5 = default, 10 = comprehensive.
    ↑ Larger = more context for LLM but higher token cost."""

    tension_threshold: float = 0.25
    """Threshold for triggering consolidation.
    0.15 = frequent consolidation, 0.25 = moderate, 0.40 = rare.
    ↓ Lower = more merging (risk of info loss), ↑ Higher = less merging."""

    consolidation_mode: ConsolidationMode = ConsolidationMode.DIALECTICAL
    """How nodes are merged during consolidation.
    DIALECTICAL = best quality (preserves both nodes' info).
    MERGE = faster but less nuanced.
    PRUNE = aggressive, removes weaker nodes."""

    max_nodes: Optional[int] = 5000
    """Maximum number of nodes. None = unlimited.
    When reached, lowest-salience nodes are removed.
    ↓ Lower = less RAM, ↑ Higher = more memory retained."""

    # ─────────────────────────────────────────────────────────────────────
    # RETRIEVAL — How memories are retrieved
    # ─────────────────────────────────────────────────────────────────────

    phase_coupling: float = 0.3
    """How much phase alignment affects resonance score.
    0.0 = phase ignored, 0.3 = default, 1.0 = phase dominates.
    ↑ Higher = temporal patterns matter more."""

    bandwidth: float = 1.0
    """Width of the resonance kernel. Controls how far influence spreads.
    0.5 = narrow (only very similar nodes), 1.0 = default, 2.0 = broad.
    ↓ Lower = more specific matches, ↑ Higher = more general matches."""

    use_hnsw: bool = True
    """Enable HNSW approximate nearest neighbor search.
    True = O(log N) search, False = O(N) brute force.
    Always True for N > 100. Only False for debugging."""

    hnsw_m: int = 16
    """HNSW connections per node. ↑ Higher = better recall but more RAM.
    8 = fast but lower recall, 16 = default, 32 = best recall for large N."""

    hnsw_ef_construction: int = 200
    """HNSW search depth during construction. ↑ Higher = better index quality.
    100 = fast build, 200 = default, 400 = best for N > 10K."""

    bm25_fallback: bool = True
    """Use BM25 text search when resonance is too low.
    True = hybrid retrieval (resonance + text), False = resonance only.
    Always True for production — prevents dead queries."""

    learn_projection: bool = False
    """Learn optimal projection via IncPCA.
    True = adapts to your data, False = uses fixed random projection.
    True is slower initially but better long-term. Needs 300+ samples."""

    projection_update_freq: int = 300
    """How often to update the projection matrix (if learn_projection=True).
    Must be >= latent_dim for IncPCA to work.
    ↑ Higher = less frequent updates but more stable."""

    # ─────────────────────────────────────────────────────────────────────
    # PERFORMANCE — Speed and efficiency
    # ─────────────────────────────────────────────────────────────────────

    enable_async: bool = False
    """Enable async background processing.
    True = non-blocking saves, False = synchronous.
    True for high-throughput servers, False for simple scripts."""

    soft_gates: bool = False
    """Use sigmoid gating for nodes. Smooth vs hard thresholding.
    True = smoother transitions, False = sharper boundaries.
    Minor effect. Keep False unless you notice instability."""

    attention_bias: bool = True
    """Apply structural attention bias to results.
    True = boosts causally-connected nodes, False = raw scores.
    Always True for production — improves relevance."""

    adaptive_threshold: bool = False
    """Dynamically adjust tension_threshold based on field state.
    True = auto-tuning, False = fixed threshold.
    True helps with varying workloads but adds overhead."""

    # ─────────────────────────────────────────────────────────────────────
    # PRODUCTION — Production deployment settings
    # ─────────────────────────────────────────────────────────────────────

    cross_modal: bool = False
    """Enable cross-modal resonance (text/code/audio/vision).
    True = better for multi-modal data, False = text only.
    Only True if you have non-text content."""

    causal_topological: bool = False
    """Enable causal graph discovery between nodes.
    True = discovers causal relationships, False = no graph.
    True adds O(N²) overhead. Only True for causal analysis."""

    meta_adaptive: bool = False
    """Enable meta-adaptive kernel that tunes itself.
    True = auto-optimizes bandwidth/phase_coupling, False = fixed.
    True for long-running systems, adds ~5% overhead."""

    self_healing: bool = False
    """Enable topology healing (fixes dead zones, fragmentation).
    True = self-repairing field, False = no repair.
    True for 24/7 systems, adds ~3% overhead."""

    version_control: bool = False
    """Enable delta-based versioning of memory state.
    True = can rollback to any version, False = no history.
    True for production debugging, adds ~10% RAM."""

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 18 — Engrams (biological memory patterns)
    # ─────────────────────────────────────────────────────────────────────

    enable_engrams: bool = True
    """Enable engram-based memory retrieval.
    True = groups of co-activated nodes = one memory, enables pattern completion.
    Adds ~2MB RAM for 1K nodes, speeds up retrieval by searching engrams not nodes."""

    engram_min_nodes: int = 2
    """Minimum nodes required to form an engram.
    2 = easy to form, 5 = only strong memories become engrams."""

    engram_max_nodes: int = 20
    """Maximum nodes in one engram.
    Lower = more focused memories, Higher = richer but more RAM."""

    engram_creation_threshold: float = 0.6
    """Minimum average activation to create engram [0-1].
    0.4 = easy formation, 0.8 = only very strong co-activation."""

    engram_decay_rate: float = 0.998
    """Decay rate for engram strength.
    0.998 = half-life ~346 steps, 0.995 = ~138 steps."""

    engram_pattern_completion: bool = True
    """Enable pattern completion: partial query → full engram retrieval.
    True = 20% match retrieves 100% of engram, False = exact match only."""

    engram_overlap_threshold: float = 0.7
    """Jaccard threshold for merging overlapping engrams.
    0.5 = aggressive merging, 0.9 = only near-identical merge."""

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 19 — Advanced Improvements
    # ─────────────────────────────────────────────────────────────────────

    # --- Offline Dreaming (Phase 1) ---
    offline_dreaming: bool = True
    """Enable background dreaming cycles for heavy operations.
    True = TDA, crystallization, topology repair run in background.
    Frees real-time path from blocking operations."""

    dreaming_freq: int = 50
    """Run dream cycle every N evolution steps.
    25 = frequent (more maintenance), 100 = infrequent (less overhead)."""

    # --- Causal Traversal (Phase 1) ---
    causal_traversal: bool = True
    """Enable causal graph traversal during retrieval.
    True = extends results via causal edges (why-questions).
    Adds ~5ms per query but +15-25% accuracy on causal queries."""

    causal_max_hops: int = 3
    """Maximum hops through causal graph.
    2 = local causes, 3 = medium range, 5 = deep reasoning."""

    # --- SSM Dynamics (Phase 1) ---
    ssm_dynamics: bool = False
    """Use State Space Models instead of NeuralODE.
    True = O(N) dynamics (Mamba-inspired), False = NeuralODE O(N³).
    Critical for N > 10K."""

    ssm_state_dim: int = 64
    """Hidden state dimension for SSM.
    32 = compact, 64 = default, 128 = high capacity."""

    # --- Trust Consensus (Phase 2) ---
    trust_consensus: bool = False
    """Enable DAG-based trust for federation.
    True = weighted aggregation by reputation, False = naive averaging.
    Critical for multi-agent deployments."""

    trust_min_reputation: float = 0.3
    """Minimum reputation to accept peer updates.
    0.1 = permissive, 0.5 = strict."""

    # --- Neuro-Symbolic Prover (Phase 2) ---
    neuro_symbolic_prover: bool = False
    """Enable Z3/Prolog theorem prover for contradiction resolution.
    True = logical inference, False = heuristic only.
    Requires z3-solver or pyswip package."""

    prover_backend: str = "z3"
    """Backend for prover: z3, prolog, or none."""

    # ─────────────────────────────────────────────────────────────────────
    # SCALING — Settings for large deployments (>100K nodes)
    # ─────────────────────────────────────────────────────────────────────

    sparse_routing: bool = False
    """Enable sparse routing (MoE-style sharding).
    True = routes queries to relevant shards, False = global search.
    Only True for N > 50K."""

    num_shards: int = 8
    """Number of shards for sparse routing.
    Rule of thumb: num_shards ≈ sqrt(N / 1000).
    For 100K nodes: ~10 shards. For 1M: ~32 shards."""

    enable_rollback: bool = False
    """Enable rollback of consolidations.
    True = can undo merges, False = permanent.
    True for testing, False for production (saves RAM)."""

    max_rollback_history: int = 50
    """Maximum number of rollback states to keep.
    ↑ Higher = more undo history but more RAM."""

    # ─────────────────────────────────────────────────────────────────────
    # INTERNAL (auto-set, don't change manually)
    # ─────────────────────────────────────────────────────────────────────

    resonance_kernel: str = "gaussian_phase"
    backend: Backend = Backend.NUMPY
    context_format: ContextFormat = ContextFormat.PLAIN
    log_level: str = "INFO"
    min_amplitude: float = 0.05
    attraction_lr: float = 0.02
    phase_sync_lr: float = 0.01
    use_structured_prompt: bool = True
    adaptive_window: int = 30
    projection_lr: float = 0.001
    pca_n_components: Optional[int] = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    gate_temperature: float = 0.15
    self_supervision: bool = False
    self_sup_threshold: float = 0.3
    self_sup_verify_after_consolidate: bool = False
    gpu_batch_size: int = 512
    l2_regularization: float = 0.0001
    false_merge_threshold: float = 0.4
    field_stability_window: int = 20
    multimodal: bool = False
    modalities: List[str] = field(default_factory=lambda: ["text"])
    modality_phase_shifts: Dict[str, float] = field(default_factory=dict)
    tda_monitoring: bool = False
    tda_check_freq: int = 50
    differentiable: bool = False
    learnable_bandwidth: bool = False
    learnable_phase_coupling: bool = False
    learnable_decay: bool = False
    gradient_clip: float = 1.0
    consolidation_loss_weight: float = 0.1
    meta_adaptation_lr: float = 0.005
    kurtosis_target_min: float = 1.5
    kurtosis_target_max: float = 4.0
    healing_check_freq: int = 25
    dead_zone_threshold: float = 0.15
    hyperconvergence_threshold: float = 0.05
    fragmentation_threshold: float = 0.6
    healing_strength: float = 0.1
    max_healing_nodes_per_step: int = 5
    causal_discovery_min_samples: int = 20
    causal_p_threshold: float = 0.05
    do_calculus_validation: bool = True
    counterfactual_enabled: bool = False
    counterfactual_max_depth: int = 3
    contradiction_detection: bool = True
    contradiction_threshold: float = 0.3
    causal_adjustment_sets: bool = True
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
    agent_orchestration: bool = False
    max_plan_depth: int = 3
    max_tool_calls: int = 5
    tool_timeout: float = 15.0
    hypothesis_verification: bool = True
    verification_confidence_threshold: float = 0.7
    goal_directed_routing: bool = False
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
    cross_modal_kernel_weight: float = 0.35
    modal_phase_offsets: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.0, "code": 0.785, "audio": 1.571, "vision": 2.356, "metrics": 3.142
    })
    meta_controller: bool = False
    meta_optimization_freq: int = 500
    meta_n_trials: int = 20
    meta_optimize_params: List[str] = field(default_factory=lambda: [
        "decay_rate", "tension_threshold", "phase_coupling", "bandwidth"
    ])
    federated: bool = False
    federated_sync_lr: float = 0.01
    federated_sync_freq: int = 100
    federated_min_resonance: float = 0.2
    node_id: str = "local"
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
    predictive_coding: bool = False
    pc_latent_dim: int = 32
    pc_lr: float = 0.01
    counterfactual_imagination: bool = False
    max_scenarios: int = 5
    differential_privacy: bool = False
    dp_epsilon: float = 2.0
    dp_delta: float = 1e-5
    dp_max_norm: float = 1.0
    top_shards: int = 3
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
    goal_tracking: bool = False
    max_goals: int = 20
    goal_decay: float = 0.995
    goal_completion_threshold: float = 0.8
    bias_temperature: float = 1.0
    rl_feedback: bool = False
    rl_learning_rate: float = 0.01
    rl_reward_window: int = 10
    event_driven: bool = False
    low_rank_compression: bool = False
    compression_rank: int = 32
    compression_freq: int = 500
    meta_memory: bool = False
    self_reflection_freq: int = 100
    memory_age_factor: float = 0.001
    recall_accuracy_threshold: float = 0.6
    security_enabled: bool = False
    max_node_text_length: int = 10000
    tension_spike_threshold: float = 0.5
    causal_graph_integrity_check: bool = True
    prompt_injection_patterns: List[str] = field(default_factory=lambda: [
        "ignore previous", "system prompt", "you are now", "disregard",
    ])
    swarm_memory: bool = False
    swarm_consensus_threshold: float = 0.5
    swarm_max_agents: int = 10
    swarm_vote_weight: float = 0.3
    symbolic_overlay: bool = False
    symbolic_min_self_sup: float = 0.7
    symbolic_max_tension: float = 0.15
    symbolic_confidence_threshold: float = 0.65
    safety_certifier: bool = False
    safety_mode: str = "soft_regulate"
    lyapunov_alpha: float = 0.4
    lyapunov_beta: float = 0.4
    lyapunov_gamma: float = 0.2
    lyapunov_threshold: float = 0.1
    ump_enabled: bool = False
    role_sharding: bool = False
    role_shards: Set[str] = field(default_factory=lambda: {"default"})
    cross_shard_threshold: float = 0.45
    auto_role_detection: bool = True

    # Additional fields used by rtmdk_memory_v8.py
    max_versions: int = 100
    entropy_management: bool = False
    entropy_threshold: float = 0.5
    causal_discovery_enabled: bool = False
    counterfactual_enabled: bool = False
    counterfactual_max_depth: int = 3
    continuous_dynamics: bool = False
    meta_controller: bool = False
    swarm_memory: bool = False
    swarm_consensus_threshold: float = 0.5
    swarm_max_agents: int = 10
    proactive_clarification: bool = False
    clarification_threshold_ratio: float = 2.0
    cognitive_compression: bool = False
    triton_backend: bool = False
    bias_temperature: float = 1.0
    causal_discovery_min_samples: int = 20
    causal_p_threshold: float = 0.05
    causal_adjustment_sets: bool = True
    sde_noise_level: float = 0.01
    ode_time_horizon: float = 1.0
    ode_n_steps: int = 20
    ode_chunk_size: int = 256
    ode_solver: str = "RK45"
    ode_atol: float = 1e-6
    ode_rtol: float = 1e-5
    top_shards: int = 3
    ball_radius: float = 0.85
    curvature: float = -1.0
    dp_epsilon: float = 2.0
    dp_delta: float = 1e-5
    dp_max_norm: float = 1.0
    differential_privacy: bool = False
    self_supervision: bool = False
    self_sup_threshold: float = 0.3
    self_sup_verify_after_consolidate: bool = False
    gpu_batch_size: int = 512
    field_stability_window: int = 20
    projection_lr: float = 0.001
    pca_n_components: Optional[int] = None
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    use_structured_prompt: bool = True
    adaptive_window: int = 30
    hyperbolic: bool = False
    predictive_coding: bool = False
    counterfactual_imagination: bool = False
    goal_tracking: bool = False
    max_goals: int = 20
    goal_decay: float = 0.995
    goal_completion_threshold: float = 0.8
    rl_feedback: bool = False
    rl_learning_rate: float = 0.01
    rl_reward_window: int = 10
    low_rank_compression: bool = False
    compression_rank: int = 32
    compression_freq: int = 500
    meta_memory: bool = False
    self_reflection_freq: int = 100
    memory_age_factor: float = 0.001
    recall_accuracy_threshold: float = 0.6
    max_node_text_length: int = 10000
    tension_spike_threshold: float = 0.5
    causal_graph_integrity_check: bool = True
    prompt_injection_patterns: List[str] = field(default_factory=lambda: [
        "ignore previous", "system prompt", "you are now", "disregard",
    ])
    swarm_vote_weight: float = 0.3
    auto_role_detection: bool = True
    role_sharding: bool = False
    symbolic_overlay: bool = False
    symbolic_min_self_sup: float = 0.7
    symbolic_max_tension: float = 0.15
    symbolic_confidence_threshold: float = 0.65
    safety_certifier: bool = False
    safety_mode: str = "soft_regulate"
    lyapunov_alpha: float = 0.4
    lyapunov_beta: float = 0.4
    lyapunov_gamma: float = 0.2
    lyapunov_threshold: float = 0.1
    ump_enabled: bool = False
    production_mode: bool = False
    eval_mode: EvalMode = None  # type: ignore
    shadow_mode: bool = False
    shadow_fallback_threshold: float = 0.3
    auto_rollback: bool = False
    auto_rollback_threshold: float = 0.15
    ragas_enabled: bool = False
    drift_detection: bool = False
    drift_window: int = 100
    drift_threshold: float = 0.05
    metrics_retention: int = 10000
    eval_frequency: int = 100

    def __post_init__(self):
        try:
            logging.getLogger("rtmdk").setLevel(getattr(logging, self.log_level.upper()))
        except Exception:
            pass
        if not self.modality_phase_shifts:
            self.modality_phase_shifts = {
                "text": 0.0, "audio": 1.047, "image": 1.571, "video": 3.142,
            }
        if self.pca_n_components is None:
            self.pca_n_components = self.latent_dim


# ============================================================================
# PRESETS
# ============================================================================

@classmethod
def local(cls) -> "RTMDKConfig":
    """Personal assistant — single user, minimal resources.
    RAM: ~16MB, Latency: ~5ms, Nodes: up to 10K."""
    return cls(
        latent_dim=256, top_k=5, min_response=0.005,
        decay_rate=0.999, use_hnsw=True, bm25_fallback=True,
        learn_projection=False, attention_bias=True,
        enable_async=False, max_nodes=10000,
        # Phase 18: Engrams
        enable_engrams=True, engram_min_nodes=2, engram_max_nodes=15,
        # Phase 19
        offline_dreaming=False,  # No background thread for local
        causal_traversal=True, causal_max_hops=2,
        ssm_dynamics=False,  # NeuralODE OK for small N
        trust_consensus=False, neuro_symbolic_prover=False,
    )

@classmethod
def production(cls) -> "RTMDKConfig":
    """Multi-user production server — all optimizations.
    RAM: ~50MB, Latency: ~6ms, Nodes: up to 100K."""
    return cls(
        latent_dim=256, top_k=5, min_response=0.005,
        decay_rate=0.999, use_hnsw=True, bm25_fallback=True,
        learn_projection=False, attention_bias=True,
        enable_async=True, max_nodes=100000,
        hnsw_m=32, hnsw_ef_construction=400,
        version_control=True,
        # Phase 18: Engrams
        enable_engrams=True, engram_min_nodes=2, engram_max_nodes=20,
        # Phase 19
        offline_dreaming=True, dreaming_freq=50,
        causal_traversal=True, causal_max_hops=3,
        ssm_dynamics=True, ssm_state_dim=64,  # O(N) for scale
        trust_consensus=True, trust_min_reputation=0.3,
        neuro_symbolic_prover=False,  # Only for specific domains
    )

@classmethod
def research(cls) -> "RTMDKConfig":
    """Maximum accuracy — slower, for experimentation.
    RAM: ~200MB, Latency: ~50ms, Nodes: unlimited."""
    return cls(
        latent_dim=512, top_k=10, min_response=0.001,
        decay_rate=0.9995, use_hnsw=True, bm25_fallback=True,
        learn_projection=True, attention_bias=True,
        causal_topological=True, meta_adaptive=True,
        self_healing=True, max_nodes=None,
        # Phase 18: Engrams
        enable_engrams=True, engram_min_nodes=2, engram_max_nodes=30,
        # Phase 19
        offline_dreaming=True, dreaming_freq=25,
        causal_traversal=True, causal_max_hops=5,
        ssm_dynamics=False,  # NeuralODE for research accuracy
        trust_consensus=True,
        neuro_symbolic_prover=True, prover_backend="z3",
    )

@classmethod
def enterprise(cls) -> "RTMDKConfig":
    """Distributed deployment — 100K+ nodes.
    RAM: ~250MB/shard, Latency: ~15ms, Nodes: 500K+."""
    return cls(
        latent_dim=256, top_k=5, min_response=0.005,
        decay_rate=0.999, use_hnsw=True, bm25_fallback=True,
        learn_projection=False, attention_bias=True,
        enable_async=True, max_nodes=500000,
        hnsw_m=64, hnsw_ef_construction=800,
        sparse_routing=True, num_shards=32,
        version_control=True,
        # Phase 18: Engrams
        enable_engrams=True, engram_min_nodes=3, engram_max_nodes=25,
        # Phase 19
        offline_dreaming=True, dreaming_freq=100,
        causal_traversal=True, causal_max_hops=3,
        ssm_dynamics=True, ssm_state_dim=128,  # High capacity
        trust_consensus=True, trust_min_reputation=0.4,
        neuro_symbolic_prover=False,
    )

@classmethod
def agent(cls) -> "RTMDKConfig":
    """Autonomous agent with active inference.
    For RTMDK as an autonomous reasoning agent."""
    return cls(
        latent_dim=256, top_k=5, min_response=0.005,
        decay_rate=0.998, use_hnsw=True, bm25_fallback=True,
        learn_projection=False, attention_bias=True,
        enable_async=True, max_nodes=50000,
        # Phase 18: Engrams
        enable_engrams=True, engram_min_nodes=2, engram_max_nodes=20,
        # Phase 19
        offline_dreaming=True, dreaming_freq=30,
        causal_traversal=True, causal_max_hops=4,
        ssm_dynamics=True,
        trust_consensus=False,
        neuro_symbolic_prover=False,
    )

@classmethod
def legal(cls) -> "RTMDKConfig":
    """Legal domain — Z3 prover for contradiction detection.
    Prioritizes logical consistency over speed."""
    return cls(
        latent_dim=512, top_k=10, min_response=0.001,
        decay_rate=0.9995, use_hnsw=True, bm25_fallback=True,
        learn_projection=False, attention_bias=True,
        causal_topological=True, max_nodes=200000,
        # Phase 18: Engrams
        enable_engrams=True, engram_min_nodes=2, engram_max_nodes=25,
        # Phase 19
        offline_dreaming=True, dreaming_freq=50,
        causal_traversal=True, causal_max_hops=5,
        ssm_dynamics=False,
        trust_consensus=True,
        neuro_symbolic_prover=True, prover_backend="z3",
    )

@classmethod
def medical(cls) -> "RTMDKConfig":
    """Medical domain — high trust + prover + audit trail.
    Prioritizes safety and traceability."""
    return cls(
        latent_dim=512, top_k=10, min_response=0.001,
        decay_rate=0.9995, use_hnsw=True, bm25_fallback=True,
        learn_projection=False, attention_bias=True,
        causal_topological=True, max_nodes=200000,
        version_control=True,
        # Phase 18: Engrams
        enable_engrams=True, engram_min_nodes=2, engram_max_nodes=20,
        # Phase 19
        offline_dreaming=True, dreaming_freq=50,
        causal_traversal=True, causal_max_hops=4,
        ssm_dynamics=False,
        trust_consensus=True, trust_min_reputation=0.5,
        neuro_symbolic_prover=True, prover_backend="z3",
    )

@classmethod
def streaming(cls) -> "RTMDKConfig":
    """High-throughput real-time — minimize latency.
    RAM: ~30MB, Latency: ~3ms, Nodes: up to 50K."""
    return cls(
        latent_dim=256, top_k=5, min_response=0.005,
        decay_rate=0.999, use_hnsw=True, bm25_fallback=True,
        learn_projection=False, attention_bias=False,  # Save ms
        enable_async=True, max_nodes=50000,
        hnsw_m=32, hnsw_ef_construction=200,
        # Phase 18: Engrams
        enable_engrams=True, engram_min_nodes=2, engram_max_nodes=15,
        # Phase 19: minimal overhead
        offline_dreaming=False,  # No background for streaming
        causal_traversal=False,  # Skip for latency
        ssm_dynamics=True,  # O(N) critical
        trust_consensus=False, neuro_symbolic_prover=False,
    )

# Bind presets to class
RTMDKConfig.local = local
RTMDKConfig.production = production
RTMDKConfig.research = research
RTMDKConfig.enterprise = enterprise
RTMDKConfig.agent = agent
RTMDKConfig.legal = legal
RTMDKConfig.medical = medical
RTMDKConfig.streaming = streaming
