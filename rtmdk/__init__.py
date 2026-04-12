"""RTMDK — Resonance-Topological Memory for LLMs.

Usage:
    from rtmdk import RTMDKMemory, RTMDKConfig

    config = RTMDKConfig.local()
    memory = RTMDKMemory(config=config, embedder=my_embedder)
"""

__version__ = "8.0.0"

from rtmdk.memory.core import RTMDKMemory, RTMDKConfig

# Re-export enums for convenience
from rtmdk.memory.core import (
    ConsolidationMode, Backend, ContextFormat,
    FieldHealth, EvalMode,
)

# Bind preset methods to RTMDKConfig class
from rtmdk.config import (
    _local, _production, _research, _enterprise,
    _agent, _legal, _medical, _streaming,
)
RTMDKConfig.local = staticmethod(_local)
RTMDKConfig.production = staticmethod(_production)
RTMDKConfig.research = staticmethod(_research)
RTMDKConfig.enterprise = staticmethod(_enterprise)
RTMDKConfig.agent = staticmethod(_agent)
RTMDKConfig.legal = staticmethod(_legal)
RTMDKConfig.medical = staticmethod(_medical)
RTMDKConfig.streaming = staticmethod(_streaming)


def list_presets():
    """List all available configuration presets."""
    return ["local", "production", "research", "enterprise",
            "agent", "legal", "medical", "streaming"]


def create_rtmdk(preset: str = "local", embedder=None) -> RTMDKMemory:
    """Create an RTMDKMemory instance with a preset configuration.

    Args:
        preset: One of 'local', 'production', 'research', 'enterprise',
                'agent', 'legal', 'medical', 'streaming'
        embedder: Embedding function (required)

    Returns:
        Configured RTMDKMemory instance
    """
    preset_methods = {
        "local": RTMDKConfig.local,
        "production": RTMDKConfig.production,
        "research": RTMDKConfig.research,
        "enterprise": RTMDKConfig.enterprise,
        "agent": RTMDKConfig.agent,
        "legal": RTMDKConfig.legal,
        "medical": RTMDKConfig.medical,
        "streaming": RTMDKConfig.streaming,
    }

    if preset not in preset_methods:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_methods.keys())}")

    config = preset_methods[preset]()
    return RTMDKMemory(config=config, embedder=embedder)


__all__ = [
    "RTMDKMemory",
    "RTMDKConfig",
    "ConsolidationMode", "Backend", "ContextFormat",
    "FieldHealth", "EvalMode",
    "list_presets",
    "create_rtmdk",
    "__version__",
]
