"""rtmdk/experimental — Research and experimental modules.

These modules are not integrated into the main RTMDK flow.
They are preserved for future research and development.
"""

# Re-export for lazy access
from .tpr import TensorProductRepresentation
from .adversarial_arena import AdversarialArena
from .active_inference import ActiveInferenceLoop

__all__ = [
    "TensorProductRepresentation",
    "AdversarialArena",
    "ActiveInferenceLoop",
]
