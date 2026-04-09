"""quick_debug.py — Quick debug script."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting debug...", flush=True)

from embedder_lmstudio import LMStudioEmbedder
emb = LMStudioEmbedder()
print(f"Available: {emb._available}", flush=True)

if emb._available:
    import numpy as np
    v = emb("hello world")
    print(f"Embedding dim: {len(v)}, norm: {np.linalg.norm(v):.2f}", flush=True)

print("Debug complete", flush=True)
