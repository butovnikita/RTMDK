"""Test all presets via factory function."""
from rtmdk import create_rtmdk, list_presets
import numpy as np

def dummy_embedder(text):
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(768).astype(np.float32) * 0.1

print("Available presets:", list(list_presets().keys()))
print()

for name in list_presets().keys():
    try:
        m = create_rtmdk(name, embedder=dummy_embedder)
        m.save_context({"input": "test", "session_id": "x"}, {"output": "ok"})
        ctx = m.load_memory_variables({"input": "test", "session_id": "x"})
        print(f"  {name:12s}: OK (nodes={len(m.field.nodes)})")
    except Exception as e:
        print(f"  {name:12s}: FAIL - {e}")

print("\nAll presets tested!")
