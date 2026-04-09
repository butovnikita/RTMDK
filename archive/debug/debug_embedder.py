"""debug_embedder.py"""
import requests, numpy as np
from embedder_lmstudio import LMStudioEmbedder

emb = LMStudioEmbedder()
print(f"Available: {emb._available}")

if emb._available:
    v1 = emb("hello world")
    print(f"Embedding dim: {len(v1)}")
    print(f"First 5 values: {v1[:5]}")

    v2 = emb("science fact number 0 with keyword fc_kw_00000")
    v3 = emb("What is the science fact number 0 keyword?")
    v4 = emb("completely unrelated topic about cooking")

    sim_23 = float(np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3) + 1e-8))
    sim_24 = float(np.dot(v2, v4) / (np.linalg.norm(v2) * np.linalg.norm(v4) + 1e-8))
    print(f"Sim(fact, query): {sim_23:.4f}")
    print(f"Sim(fact, random): {sim_24:.4f}")

    # Now test RTMDK
    from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory

    config = RTMDKConfig(
        embedding_dim=768, latent_dim=128, top_k=5, min_response=0.001,
        decay_rate=0.999, enable_async=False, bm25_fallback=True,
        use_hnsw=False, learn_projection=False,
    )
    mem = RTMDKMemory(config=config, embedder=emb)

    fact = "science fact number 0 with unique keyword fc_kw_00000"
    mem.save_context({"input": fact, "session_id": "t"}, {"output": fact})
    print(f"Nodes after store: {len(mem.field.nodes)}")

    ctx = mem.load_memory_variables({"input": "science fact number 0", "session_id": "t"})
    context = ctx["rtmdk_context"]
    print(f"Context length: {len(context)}")
    print(f"Contains fc_kw_00000: {'fc_kw_00000' in context.lower()}")
    print(f"Context[:200]: {context[:200]}")
    print("DEBUG DONE")
