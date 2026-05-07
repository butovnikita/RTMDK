"""
rtmdk/utils/preset_recommender.py — Auto-select Best Config Preset.

Recommends optimal preset based on user requirements.
"""

from typing import Any, Dict


def recommend_preset(
    expected_nodes: int = 1000,
    available_ram_mb: float = 256,
    max_latency_ms: float = 100,
    use_case: str = "general",
) -> Dict[str, Any]:
    """Recommend optimal RTMDK preset.

    Args:
        expected_nodes: Expected number of memory nodes
        available_ram_mb: Available RAM in MB
        max_latency_ms: Maximum acceptable latency
        use_case: 'general', 'chat', 'search', 'agent', 'legal', 'medical'

    Returns:
        Dict with recommended preset name and config overrides
    """
    # Decision logic
    if expected_nodes <= 10000 and available_ram_mb >= 16:
        preset = "local"
    elif expected_nodes <= 50000 and available_ram_mb >= 30:
        preset = "streaming" if max_latency_ms < 50 else "agent"
    elif expected_nodes <= 100000 and available_ram_mb >= 50:
        preset = "production"
    elif expected_nodes <= 200000 and use_case == "legal":
        preset = "legal"
    elif expected_nodes <= 200000 and use_case == "medical":
        preset = "medical"
    elif expected_nodes <= 500000:
        preset = "enterprise"
    else:
        preset = "enterprise"

    # Use-case overrides
    if use_case == "research":
        preset = "research"
    elif use_case == "agent":
        preset = "agent"

    overrides = {}
    if max_latency_ms < 20:
        overrides["offline_dreaming"] = False
        overrides["causal_traversal"] = False
        overrides["attention_bias"] = False

    return {
        "preset": preset,
        "overrides": overrides,
        "estimated_ram_mb": _estimate_ram(expected_nodes),
        "estimated_latency_ms": _estimate_latency(expected_nodes, overrides),
    }


def _estimate_ram(nodes: int) -> float:
    """Estimate RAM usage for given node count."""
    base = 10  # Base overhead MB
    per_node = 0.002  # ~2KB per node
    return round(base + nodes * per_node, 1)


def _estimate_latency(nodes: int, overrides: Dict) -> float:
    """Estimate latency for given node count."""
    base_ms = 5
    per_1000_nodes = 0.05
    latency = base_ms + (nodes / 1000) * per_1000_nodes
    if overrides.get("causal_traversal") is False:
        latency *= 0.8
    if overrides.get("attention_bias") is False:
        latency *= 0.9
    return round(latency, 1)
