"""RTMDK — Resonance-Topological Memory for LLMs.

Usage:
    from rtmdk import RTMDKMemory, RTMDKConfig
    
    config = RTMDKConfig.local()
    memory = RTMDKMemory(config=config, embedder=my_embedder)
"""

__version__ = "8.0.0"

from rtmdk.memory.core import RTMDKMemory, RTMDKConfig
from rtmdk.config import RTMDKConfig as ConfigProfiles

# Export config presets for easy access
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
        "local": ConfigProfiles.local,
        "production": ConfigProfiles.production,
        "research": ConfigProfiles.research,
        "enterprise": ConfigProfiles.enterprise,
        "agent": ConfigProfiles.agent,
        "legal": ConfigProfiles.legal,
        "medical": ConfigProfiles.medical,
        "streaming": ConfigProfiles.streaming,
    }
    
    if preset not in preset_methods:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_methods.keys())}")
    
    config = preset_methods[preset]()
    return RTMDKMemory(config=config, embedder=embedder)


__all__ = [
    "RTMDKMemory",
    "RTMDKConfig", 
    "ConfigProfiles",
    "list_presets",
    "create_rtmdk",
    "__version__",
]
