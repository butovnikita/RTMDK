"""
rtmdk_memory_v8.py — Compatibility Shim.

This file provides backward compatibility for scripts that import from the old location.
All code has been moved to rtmdk/memory/core.py
"""

from rtmdk.memory.core import (
    RTMDKMemory,
    RTMDKConfig,
    RTMDKField,
    ContextFormat,
    apply_attention_bias,
    format_cognitive_context,
    detect_modality,
    detect_tier,
    MemoryNode,
)

# Re-export everything for backward compatibility
__all__ = [
    "RTMDKMemory",
    "RTMDKConfig",
    "RTMDKField",
    "ContextFormat",
    "apply_attention_bias",
    "format_cognitive_context",
    "detect_modality",
    "detect_tier",
    "MemoryNode",
]
