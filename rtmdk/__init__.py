"""RTMDK — Resonance-Topological Memory for LLMs v8.0

Quick Start:
    from rtmdk import create_rtmdk
    
    # Create with preset
    memory = create_rtmdk("production", embedder=my_embedder)
    
    # Or create manually
    from rtmdk import RTMDKConfig, RTMDKMemory
    config = RTMDKConfig.local()
    memory = RTMDKMemory(config=config, embedder=my_embedder)

Available presets: local, production, research, enterprise, agent, legal, medical, streaming
"""

import os
import sys
import importlib

# ─────────────────────────────────────────────────────────────────
# Core: Config and monolith (always available)
# ─────────────────────────────────────────────────────────────────
from rtmdk.config import (
    RTMDKConfig,
    ConsolidationMode,
    Backend,
    ContextFormat,
    FieldHealth,
    EvalMode,
)

_monolith_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _monolith_dir not in sys.path:
    sys.path.insert(0, _monolith_dir)
from rtmdk_memory_v8 import RTMDKField, RTMDKMemory

# ─────────────────────────────────────────────────────────────────
# Factory function — easiest way to create RTMDK
# ─────────────────────────────────────────────────────────────────

def create_rtmdk(preset: str = "production", embedder=None, **kwargs) -> RTMDKMemory:
    """Create an RTMDKMemory instance with a preset configuration.
    
    Args:
        preset: One of 'local', 'production', 'research', 'enterprise',
                'agent', 'legal', 'medical', 'streaming'
        embedder: Callable that takes text and returns np.ndarray
        **kwargs: Override any config parameter
    
    Returns:
        RTMDKMemory instance ready to use
    """
    # Use the monolith's own RTMDKConfig to avoid type mismatch
    from rtmdk_memory_v8 import RTMDKConfig as MonoConfig
    
    preset_defaults = {
        "local": dict(latent_dim=256, top_k=5, decay_rate=0.999, max_nodes=10000,
                      enable_engrams=True, offline_dreaming=False, causal_traversal=True,
                      causal_max_hops=2, ssm_dynamics=False, trust_consensus=False,
                      neuro_symbolic_prover=False, learn_projection=False),
        "production": dict(latent_dim=256, top_k=5, decay_rate=0.999, max_nodes=100000,
                           enable_engrams=True, offline_dreaming=True, dreaming_freq=50,
                           causal_traversal=True, causal_max_hops=3, ssm_dynamics=True,
                           ssm_state_dim=64, trust_consensus=True, hnsw_m=32,
                           hnsw_ef_construction=400, enable_async=True),
        "research": dict(latent_dim=512, top_k=10, decay_rate=0.9995, max_nodes=None,
                         enable_engrams=True, offline_dreaming=True, dreaming_freq=25,
                         causal_traversal=True, causal_max_hops=5, neuro_symbolic_prover=True,
                         prover_backend="z3", learn_projection=True),
        "enterprise": dict(latent_dim=256, top_k=5, decay_rate=0.999, max_nodes=500000,
                           enable_engrams=True, offline_dreaming=True, dreaming_freq=100,
                           causal_traversal=True, ssm_dynamics=True, ssm_state_dim=128,
                           trust_consensus=True, hnsw_m=64, hnsw_ef_construction=800,
                           sparse_routing=True, num_shards=32, enable_async=True),
        "agent": dict(latent_dim=256, top_k=5, decay_rate=0.998, max_nodes=50000,
                      enable_engrams=True, offline_dreaming=True, dreaming_freq=30,
                      causal_traversal=True, causal_max_hops=4, ssm_dynamics=True),
        "legal": dict(latent_dim=512, top_k=10, decay_rate=0.9995, max_nodes=200000,
                      enable_engrams=True, offline_dreaming=True, causal_traversal=True,
                      causal_max_hops=5, neuro_symbolic_prover=True, prover_backend="z3",
                      trust_consensus=True),
        "medical": dict(latent_dim=512, top_k=10, decay_rate=0.9995, max_nodes=200000,
                        enable_engrams=True, offline_dreaming=True, causal_traversal=True,
                        causal_max_hops=4, neuro_symbolic_prover=True, trust_consensus=True,
                        trust_min_reputation=0.5, version_control=True),
        "streaming": dict(latent_dim=256, top_k=5, decay_rate=0.999, max_nodes=50000,
                          enable_engrams=True, offline_dreaming=False, causal_traversal=False,
                          ssm_dynamics=True, attention_bias=False, hnsw_m=32,
                          hnsw_ef_construction=200, enable_async=True),
    }
    
    if preset not in preset_defaults:
        raise ValueError(f"Unknown preset: '{preset}'. Available: {list(preset_defaults.keys())}")
    
    params = preset_defaults[preset]
    for key, value in kwargs.items():
        params[key] = value
    
    if embedder is None:
        raise ValueError("embedder is required")
    
    config = MonoConfig(**params)
    return RTMDKMemory(config=config, embedder=embedder)


def list_presets() -> dict:
    """List available presets with their key settings."""
    result = {}
    for name, factory in [
        ("local", RTMDKConfig.local),
        ("production", RTMDKConfig.production),
        ("research", RTMDKConfig.research),
        ("enterprise", RTMDKConfig.enterprise),
        ("agent", RTMDKConfig.agent),
        ("legal", RTMDKConfig.legal),
        ("medical", RTMDKConfig.medical),
        ("streaming", RTMDKConfig.streaming),
    ]:
        cfg = factory()
        result[name] = {
            "latent_dim": cfg.latent_dim,
            "top_k": cfg.top_k,
            "decay_rate": cfg.decay_rate,
            "engrams": cfg.enable_engrams,
            "dreaming": cfg.offline_dreaming,
            "causal": cfg.causal_traversal,
            "ssm": cfg.ssm_dynamics,
        }
    return result


# ─────────────────────────────────────────────────────────────────
# Lazy imports — loaded on demand to avoid heavy startup cost
# ─────────────────────────────────────────────────────────────────

def __getattr__(name):
    """Lazy load modules on first access."""
    _lazy_imports = {
        # Nodes
        "MemoryNode": ("rtmdk.nodes", ["MemoryNode"]),
        "CausalEdge": ("rtmdk.nodes", ["CausalEdge"]),
        "EvalResult": ("rtmdk.nodes", ["EvalResult"]),
        "FederatedNode": ("rtmdk.nodes", ["FederatedNode"]),
        "GoalNode": ("rtmdk.nodes", ["GoalNode"]),
        # Utils
        "detect_modality": ("rtmdk.utils.modality", ["detect_modality"]),
        "detect_tier": ("rtmdk.utils.modality", ["detect_tier"]),
        "poincare_dist": ("rtmdk.utils.hyperbolic", ["poincare_dist"]),
        "apply_attention_bias": ("rtmdk.utils.attention", ["apply_attention_bias"]),
        "format_context": ("rtmdk.utils.formatting", ["format_context"]),
        "build_system_prompt": ("rtmdk.utils.formatting", ["build_system_prompt"]),
        # Engrams (Phase 18)
        "EngramPattern": ("rtmdk.engrams", ["EngramPattern"]),
        "EngramIndex": ("rtmdk.engrams", ["EngramIndex"]),
        "EngramManager": ("rtmdk.engrams", ["EngramManager"]),
        "PatternCompleter": ("rtmdk.engrams", ["PatternCompleter"]),
        # Phase 19
        "OfflineDreamer": ("rtmdk.production.offline_dreamer", ["OfflineDreamer"]),
        "CausalTraversalEngine": ("rtmdk.engines.causal_traversal", ["CausalTraversalEngine"]),
        "SSMDynamics": ("rtmdk.engines.ssm_dynamics", ["SSMDynamics"]),
        "TrustConsensusEngine": ("rtmdk.engines.trust_consensus", ["TrustConsensusEngine"]),
        "NeuroSymbolicProver": ("rtmdk.engines.neuro_symbolic_prover", ["NeuroSymbolicProver"]),
        "ActiveInferenceLoop": ("rtmdk.production.active_inference", ["ActiveInferenceLoop"]),
        "AdversarialArena": ("rtmdk.production.adversarial_arena", ["AdversarialArena"]),
        # Config enums
        "ConsolidationMode": ("rtmdk.config", ["ConsolidationMode"]),
        "Backend": ("rtmdk.config", ["Backend"]),
        "ContextFormat": ("rtmdk.config", ["ContextFormat"]),
    }
    
    if name in _lazy_imports:
        module_path, names = _lazy_imports[name]
        mod = importlib.import_module(module_path)
        for n in names:
            globals()[n] = getattr(mod, n)
        return globals()[name]
    
    raise AttributeError(f"module 'rtmdk' has no attribute '{name}'")


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────
__all__ = [
    "create_rtmdk",
    "list_presets",
    "RTMDKConfig",
    "RTMDKMemory",
    "RTMDKField",
    "ConsolidationMode",
    "Backend",
    "ContextFormat",
    "FieldHealth",
    "EvalMode",
    "MemoryNode",
    "CausalEdge",
    "EngramPattern",
    "EngramIndex",
    "EngramManager",
    "OfflineDreamer",
    "CausalTraversalEngine",
    "SSMDynamics",
    "TrustConsensusEngine",
    "NeuroSymbolicProver",
    "detect_modality",
    "format_context",
]
