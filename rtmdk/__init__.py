"""
RTMDK — Resonance-Topological Memory for LLMs.

Production-ready long-term memory system with resonance-based retrieval.

Usage:
    from rtmdk import RTMDKMemory, RTMDKConfig
    from rtmdk.server import create_app
    from rtmdk.proxy import create_proxy_app
"""

__version__ = "8.0.0"
__author__ = "RTMDK Team"

# Core exports
from rtmdk.memory.core import RTMDKMemory, RTMDKConfig

__all__ = [
    "RTMDKMemory",
    "RTMDKConfig",
]
