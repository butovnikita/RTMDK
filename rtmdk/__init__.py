"""RTMDK — Resonance-Topological Memory for LLMs.

Usage:
    from rtmdk import RTMDKMemory, RTMDKConfig

    config = RTMDKConfig.local()
    memory = RTMDKMemory(config=config, embedder=my_embedder)
"""

from rtmdk.nodes import MemoryNode

__version__ = "8.3.2"

from rtmdk.memory.config import RTMDKConfig, ConsolidationMode, Backend, ContextFormat, FieldHealth, EvalMode
from rtmdk.memory.core import RTMDKMemory, RTMDKField

# Bind preset methods to RTMDKConfig class
from rtmdk.config import (
    _local,
    _production,
    _research,
    _enterprise,  # type: ignore[attr-defined]
    _agent,
    _legal,
    _medical,
    _streaming,
    _sillytavern,  # type: ignore[attr-defined]
)

RTMDKConfig.local = staticmethod(_local)  # type: ignore
RTMDKConfig.production = staticmethod(_production)  # type: ignore
RTMDKConfig.research = staticmethod(_research)  # type: ignore
RTMDKConfig.enterprise = staticmethod(_enterprise)  # type: ignore
RTMDKConfig.agent = staticmethod(_agent)  # type: ignore
RTMDKConfig.legal = staticmethod(_legal)  # type: ignore
RTMDKConfig.medical = staticmethod(_medical)  # type: ignore
RTMDKConfig.streaming = staticmethod(_streaming)  # type: ignore
RTMDKConfig.sillytavern = staticmethod(_sillytavern)  # type: ignore


def list_presets():
    """List all available configuration presets."""
    return ["local", "production", "research", "enterprise", "agent", "legal", "medical", "streaming", "sillytavern"]


def create_rtmdk(preset: str = "local", embedder=None) -> RTMDKMemory:
    """Create an RTMDKMemory instance with a preset configuration.

    Args:
        preset: One of 'local', 'production', 'research', 'enterprise',
                'agent', 'legal', 'medical', 'streaming', 'sillytavern'
        embedder: Embedding function (required)

    Returns:
        Configured RTMDKMemory instance
    """
    preset_methods = {
        "local": RTMDKConfig.local,  # type: ignore
        "production": RTMDKConfig.production,  # type: ignore
        "research": RTMDKConfig.research,  # type: ignore
        "enterprise": RTMDKConfig.enterprise,  # type: ignore
        "agent": RTMDKConfig.agent,  # type: ignore
        "legal": RTMDKConfig.legal,  # type: ignore
        "medical": RTMDKConfig.medical,  # type: ignore
        "streaming": RTMDKConfig.streaming,  # type: ignore
        "sillytavern": RTMDKConfig.sillytavern,  # type: ignore
    }

    if preset not in preset_methods:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_methods.keys())}")

    config = preset_methods[preset]()
    return RTMDKMemory(config=config, embedder=embedder)


__all__ = [
    "RTMDKMemory",
    "RTMDKConfig",
    "RTMDKField",
    "MemoryNode",
    "ConsolidationMode",
    "Backend",
    "ContextFormat",
    "FieldHealth",
    "EvalMode",
    "list_presets",
    "create_rtmdk",
    "__version__",
]
