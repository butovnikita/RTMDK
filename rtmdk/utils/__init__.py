"""rtmdk/utils/__init__.py"""

from .modality import detect_modality, detect_tier
from .hyperbolic import poincare_dist, exp_map_poincare, log_map_poincare, mobius_add
from .attention import apply_attention_bias, format_cognitive_context
from .formatting import format_context, build_system_prompt, SYSTEM_PROMPT_TEMPLATES

__all__ = [
    "detect_modality",
    "detect_tier",
    "poincare_dist",
    "exp_map_poincare",
    "log_map_poincare",
    "mobius_add",
    "apply_attention_bias",
    "format_cognitive_context",
    "format_context",
    "build_system_prompt",
    "SYSTEM_PROMPT_TEMPLATES",
]
