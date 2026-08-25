"""
rtmdk/config.py — Configuration Presets for RTMDK.

R3.2 (2026-08-24, audit/risks-2026-08-24): single source of truth is
rtmdk/memory/config.py (230+ flat fields, ORPHANED_FLAGS, validate).
This module is a thin re-export + presets only — no duplicated defaults
beyond explicit preset overrides (see BACKLOG.md R3.1). Importing
RTMDKConfig from here or from rtmdk.memory.config is equivalent, but
memory.config is canonical for class definition. Presets delegate to
RTMDKConfig(**kwargs) and are bound in rtmdk/__init__.py.

Usage:
    from rtmdk.config import RTMDKConfig  # re-export, same as from rtmdk.memory.config
    from rtmdk.memory.config import RTMDKConfig  # canonical
    config = RTMDKConfig.local()           # Minimal resources
    config = RTMDKConfig.production()      # Full optimizations
    config = RTMDKConfig.research()        # Maximum accuracy

Presets:
    local()       — Single user, minimal resources (~16MB, 10K nodes)
    production()  — Multi-user, all optimizations (~50MB, 100K nodes)
    research()    — Maximum accuracy, slower (~200MB, unlimited)
    enterprise()  — Distributed, sharding (~250MB/shard, 500K+ nodes)
    agent()       — Autonomous agent with active inference
    legal()       — Legal domain, Z3 prover for contradictions
    medical()     — Medical domain, high trust + audit trail
    streaming()   — High-throughput real-time (~3ms, 50K nodes)
"""

# Re-export the single source of truth — thin wrapper, no duplication (R3.2)
# Canonical definition lives in rtmdk/memory/config.py; this re-export keeps
# backward compat for `from rtmdk.config import RTMDKConfig`.
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.config import ConsolidationMode, Backend, ContextFormat, FieldHealth, EvalMode  # noqa: F401

# Re-export enums


def _local() -> RTMDKConfig:
    """Personal assistant — single user, minimal resources."""
    from rtmdk.memory.core import ContextFormat

    return RTMDKConfig(
        latent_dim=256,
        top_k=5,
        decay_rate=0.999,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=False,
        projection_mode="identity",
        hybrid_alpha=0.7,
        attention_bias=True,
        enable_async=False,
        max_nodes=10000,
        context_format=ContextFormat.ATTENTION,
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=15,
        offline_dreaming=False,
        causal_traversal=True,
        causal_max_hops=2,
        ssm_dynamics=False,
        trust_consensus=False,
        neuro_symbolic_prover=False,
    )


def _production() -> RTMDKConfig:
    """Multi-user production server — all optimizations. R12.2: requires RTMDK_API_KEY != rtmdk-local."""
    return RTMDKConfig(
        production_mode=True,  # R12.2: enables api_key validation (ERROR if rtmdk-local)
        latent_dim=256,
        top_k=5,
        min_response=0.001,
        decay_rate=0.999,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=False,
        projection_mode="identity",
        hybrid_alpha=0.7,
        attention_bias=True,
        enable_async=True,
        max_nodes=100000,
        hnsw_m=32,
        hnsw_ef_construction=400,
        version_control=True,
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=20,
        offline_dreaming=True,
        dreaming_freq=50,
        causal_traversal=True,
        causal_max_hops=3,
        ssm_dynamics=True,
        ssm_state_dim=64,
        trust_consensus=True,
        trust_min_reputation=0.3,
        neuro_symbolic_prover=False,
        # Performance & accuracy modules (previously disabled)
        conformal_prediction=True,
        conformal_alpha=0.10,
        conformal_min_calib=50,
        meta_adaptive=True,
        # Adaptive bandwidth via MetaAdaptiveKernel (global kurtosis-based)
        enable_kalman_filter=True,
        kalman_diagonal_approx=True,
        # Accuracy tuning (post-benchmark optimization)
        resonance_kernel="cosine",
        # Phase 21: SOT enabled by default for out-of-the-box autonomy
        sot_enabled=True,
        sot_use_for_query=True,
        sot_tokenization_mode="word",
        sot_attention_pooling=True,
        sot_subword_seed=True,
        sot_max_vocab=4096,
        sot_contrastive_lr=0.01,
        sot_ssm_sync=True,
        sot_diagonal_ssm=True,
        sot_retrieval_feedback=True,
        sot_hard_negatives=True,
        sot_merge_freq=100,
    )


def _research() -> RTMDKConfig:
    """Maximum accuracy — slower, for experimentation."""
    return RTMDKConfig(
        latent_dim=512,
        top_k=10,
        min_response=0.001,
        decay_rate=0.9995,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=True,
        attention_bias=True,
        causal_topological=True,
        meta_adaptive=True,
        self_healing=True,
        max_nodes=None,
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=30,
        offline_dreaming=True,
        dreaming_freq=25,
        causal_traversal=True,
        causal_max_hops=5,
        ssm_dynamics=False,
        trust_consensus=True,
        neuro_symbolic_prover=True,
        prover_backend="z3",
    )


def _enterprise() -> RTMDKConfig:
    """Distributed deployment — 100K+ nodes."""
    return RTMDKConfig(
        latent_dim=256,
        top_k=5,
        decay_rate=0.999,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=False,
        projection_mode="identity",
        hybrid_alpha=0.7,
        attention_bias=True,
        meta_adaptive=True,
        enable_async=True,
        max_nodes=500000,
        hnsw_m=64,
        hnsw_ef_construction=800,
        sparse_routing=True,
        num_shards=32,
        version_control=True,
        enable_engrams=True,
        engram_min_nodes=3,
        engram_max_nodes=25,
        offline_dreaming=True,
        dreaming_freq=100,
        causal_traversal=True,
        causal_max_hops=3,
        ssm_dynamics=True,
        ssm_state_dim=128,
        trust_consensus=True,
        trust_min_reputation=0.4,
        neuro_symbolic_prover=False,
    )


def _agent() -> RTMDKConfig:
    """Autonomous agent with active inference."""
    return RTMDKConfig(
        latent_dim=256,
        top_k=5,
        decay_rate=0.998,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=False,
        projection_mode="identity",
        hybrid_alpha=0.7,
        attention_bias=True,
        meta_adaptive=True,
        enable_async=True,
        max_nodes=50000,
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=20,
        offline_dreaming=True,
        dreaming_freq=30,
        causal_traversal=True,
        causal_max_hops=4,
        ssm_dynamics=True,
        trust_consensus=False,
        neuro_symbolic_prover=False,
    )


def _legal() -> RTMDKConfig:
    """Legal domain — Z3 prover for contradiction detection."""
    return RTMDKConfig(
        latent_dim=512,
        top_k=10,
        min_response=0.001,
        decay_rate=0.9995,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=False,
        attention_bias=True,
        causal_topological=True,
        max_nodes=200000,
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=25,
        offline_dreaming=True,
        dreaming_freq=50,
        causal_traversal=True,
        causal_max_hops=5,
        ssm_dynamics=False,
        trust_consensus=True,
        neuro_symbolic_prover=True,
        prover_backend="z3",
        domain_aware_retrieval=True,
        domain_consolidation_guard=True,
    )


def _medical() -> RTMDKConfig:
    """Medical domain — high trust + prover + audit trail."""
    return RTMDKConfig(
        latent_dim=512,
        top_k=10,
        min_response=0.001,
        decay_rate=0.9995,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=False,
        attention_bias=True,
        causal_topological=True,
        max_nodes=200000,
        version_control=True,
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=20,
        offline_dreaming=True,
        dreaming_freq=50,
        causal_traversal=True,
        causal_max_hops=4,
        ssm_dynamics=False,
        trust_consensus=True,
        trust_min_reputation=0.5,
        neuro_symbolic_prover=True,
        prover_backend="z3",
        domain_aware_retrieval=True,
        domain_consolidation_guard=True,
    )


def _streaming() -> RTMDKConfig:
    """High-throughput real-time — minimize latency."""
    return RTMDKConfig(
        latent_dim=256,
        top_k=5,
        decay_rate=0.999,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=False,
        projection_mode="identity",
        hybrid_alpha=0.7,
        attention_bias=False,
        meta_adaptive=True,
        enable_async=True,
        max_nodes=50000,
        hnsw_m=32,
        hnsw_ef_construction=200,
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=15,
        offline_dreaming=False,
        causal_traversal=False,
        ssm_dynamics=True,
        trust_consensus=False,
        neuro_symbolic_prover=False,
    )


def _sillytavern() -> RTMDKConfig:
    """SillyTavern — no system prompt, ST manages prompts via character cards."""
    cfg = RTMDKConfig(
        latent_dim=64,
        top_k=5,
        decay_rate=0.997,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=True,
        attention_bias=True,
        enable_async=False,
        max_nodes=10000,
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=15,
        offline_dreaming=False,
        causal_traversal=True,
        causal_max_hops=2,
        ssm_dynamics=False,
        trust_consensus=False,
        neuro_symbolic_prover=False,
        system_prompt=None,  # ← Key difference: no system prompt
    )
    return cfg


def _benchmark() -> RTMDKConfig:
    """Benchmark preset — maximizes recall@K on standard QA datasets."""
    return RTMDKConfig(
        embedding_dim=384,
        latent_dim=384,
        top_k=5,
        min_response=0.001,
        decay_rate=0.999,
        use_hnsw=True,
        bm25_fallback=True,
        learn_projection=False,
        projection_mode="identity",
        hybrid_alpha=0.7,
        attention_bias=True,
        meta_adaptive=True,
        enable_async=True,
        max_nodes=100000,
        hnsw_m=32,
        hnsw_ef_construction=400,
        resonance_kernel="cosine",
        enable_engrams=True,
        engram_min_nodes=2,
        engram_max_nodes=15,
        query_cache_size=1000,
        query_cache_ttl=3600,
        conformal_prediction=True,
        conformal_alpha=0.10,
        conformal_min_calib=50,
        offline_dreaming=False,
        causal_traversal=True,
        causal_max_hops=3,
        ssm_dynamics=False,
        trust_consensus=False,
        neuro_symbolic_prover=False,
        # P1 Differentiation features
        contextual_retrieval=True,
        contextual_backend="heuristic",
        bgem3_enabled=False,  # Toggle to True when BGE-M3 embedder is used
        cascade_enabled=True,
        cascade_causal_threshold=0.3,
        cascade_factual_threshold=0.3,
        # SOT defaults for best quality
        sot_enabled=False,
        sot_tokenization_mode="word",
        sot_max_vocab=10000,
        sot_attention_pooling=True,
    )


# Bind presets as class methods for backward compatibility
RTMDKConfig.local = staticmethod(_local)  # type: ignore
RTMDKConfig.production = staticmethod(_production)  # type: ignore
RTMDKConfig.research = staticmethod(_research)  # type: ignore
RTMDKConfig.enterprise = staticmethod(_enterprise)  # type: ignore
RTMDKConfig.agent = staticmethod(_agent)  # type: ignore
RTMDKConfig.legal = staticmethod(_legal)  # type: ignore
RTMDKConfig.medical = staticmethod(_medical)  # type: ignore
RTMDKConfig.streaming = staticmethod(_streaming)  # type: ignore
RTMDKConfig.sillytavern = staticmethod(_sillytavern)  # type: ignore
RTMDKConfig.benchmark = staticmethod(_benchmark)  # type: ignore
