# Shared pytest fixtures and configuration for RTMDK tests
import os
import sys

# Legacy SillyTavern modules (embedder_lmstudio, rtmdk_server_ux, ...)
# live in <root>/legacy after the repo cleanup; make them importable.
_LEGACY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "legacy")
if os.path.isdir(_LEGACY_DIR) and _LEGACY_DIR not in sys.path:
    sys.path.insert(0, _LEGACY_DIR)
