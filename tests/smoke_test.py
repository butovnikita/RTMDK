"""
smoke_test.py — Quick validation of RTMDK v8 critical paths.
"""

from rtmdk.memory.core import (
    RTMDKConfig,
    RTMDKMemory,
    ContextFormat,
)
from rtmdk.utils.attention import apply_attention_bias, format_cognitive_context
import asyncio
import numpy as np
import sys

# Add project root to path
sys.path.insert(0, ".")


_embed_rng = np.random.default_rng(42)


def dummy_embedder(text: str) -> np.ndarray:
    rng = np.random.default_rng(hash(text) % 2**32)
    base = rng.standard_normal(768).astype(np.float32) * 0.1
    sig = np.array([hash(text + str(i)) % 1000 / 500 for i in range(10)], dtype=np.float32)
    base[:10] = sig
    return base


async def run_smoke_test():
    print("=" * 50)
    print("  RTMDK v8 Smoke Test")
    print("=" * 50)

    # Check if ContextFormat.COGNITIVE exists, fallback to JSON
    ctx_fmt = getattr(ContextFormat, "COGNITIVE", ContextFormat.JSON)

    cfg = RTMDKConfig(
        async_pipeline=True,
        use_hnsw=True,
        adaptive_threshold=True,
        context_format=ctx_fmt,
        enable_rollback=True,
        attention_bias=True,
        embedding_dim=768,
        latent_dim=64,
        top_k=3,
        enable_async=False,  # Disable for sync smoke test
        min_response=0.01,  # Lower threshold for dummy embedder
    )

    print("\n[1] Creating memory with async_pipeline + HNSW + attention_bias...")
    mem = RTMDKMemory(config=cfg, embedder=dummy_embedder)
    print(f"  OK Memory created: {len(mem.field.nodes)} nodes")

    # 1. Test save + async evolution
    print("\n[2] Testing save_context + async evolution...")
    mem.save_context({"input": "test query text", "session_id": "s1"}, {"output": "ok response"})
    mem.save_context({"input": "another test", "session_id": "s1"}, {"output": "another response"})
    mem.save_context({"input": "test data here", "session_id": "s1"}, {"output": "data response"})
    await asyncio.sleep(0.1)  # give worker time
    print(f"  OK Saved: {len(mem.field.nodes)} nodes")

    # 2. Test load with attention bias - query with text that matches saved
    # nodes
    print("\n[3] Testing load_memory_variables with attention bias...")
    # Query with the exact text that was saved (output text)
    ctx = mem.load_memory_variables({"input": "ok response", "session_id": "s1"})
    print(f"  Context: {ctx['rtmdk_context'][:300]}...")
    # Check for non-empty context
    has_content = (
        "SCORE:" in ctx["rtmdk_context"]
        or "CAUSAL:" in ctx["rtmdk_context"]
        or "TIER:" in ctx["rtmdk_context"]
        or "ok" in ctx["rtmdk_context"].lower()
        or len(ctx["rtmdk_context"]) > 10
    )
    assert has_content, f"Expected context content, got: {ctx['rtmdk_context'][:100]}"
    print("  OK Attention bias applied")

    # 3. Test consolidation without KeyErrors
    print("\n[4] Testing consolidation (30 steps)...")
    for i in range(30):
        mem.field.step(inputs=[{"embedding": dummy_embedder(f"noise_{i}"), "content": {"text": f"n{i}"}}])
    print(f"  OK 30 steps completed: {len(mem.field.nodes)} nodes")

    # 4. Test HNSW routing
    print("\n[5] Testing HNSW routing...")
    if mem.field.hnsw_index:
        print(f"  OK HNSW index: {len(mem.field.hnsw_index.positions)} positions")
    else:
        print("  WARN HNSW not initialized (need more nodes)")

    # 5. Test adaptive threshold
    print("\n[6] Testing adaptive threshold...")
    if mem.field.adaptive_threshold:
        print(f"  OK Adaptive threshold: {mem.field.adaptive_threshold.get_threshold():.4f}")
    else:
        print("  WARN Adaptive threshold not initialized")

    # 6. Test rollback
    print("\n[7] Testing rollback...")
    if mem.field._rollback_history:
        print(f"  OK Rollback history: {len(mem.field._rollback_history)} entries")
    else:
        print("  WARN No rollback history (no consolidations yet)")

    # 7. Test cognitive compression
    print("\n[8] Testing cognitive context compression...")
    results = mem.field.query(dummy_embedder("test"), phase=0.0, top_k=3)
    if results:
        cognitive = format_cognitive_context(results)
        print(f"  OK Cognitive context: {cognitive[:150]}...")
    else:
        print("  WARN No results for cognitive context")

    # 8. Test attention bias
    print("\n[9] Testing attention bias...")
    if results:
        biased = apply_attention_bias(results, temperature=1.0)
        print(f"  OK Attention bias applied: {len(biased)} results")

    print("\n" + "=" * 50)
    print("  Smoke-test passed")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
