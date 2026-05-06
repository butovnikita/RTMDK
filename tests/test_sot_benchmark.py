"""
Quick SOT benchmark: compare SOT query vs external embedder on 100 QA records.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"


def run_sot_benchmark():
    # Load data
    with open("datasets/qa_1000_en.json", "r", encoding="utf-8") as f:
        data = json.load(f)["records"]

    # External embedder (dummy for fast test — SOT will do the work)
    def dummy_embed(text: str) -> np.ndarray:
        h = hash(text) % (2**31)
        rng = np.random.RandomState(h)
        e = rng.randn(768).astype(np.float32)
        return e / np.linalg.norm(e)

    # Config with SOT enabled
    cfg = RTMDKConfig(
        embedding_dim=768,
        latent_dim=128,
        top_k=15,
        min_response=0.001,
        decay_rate=0.999,
        use_hnsw=False,
        learn_projection=False,
        bm25_fallback=False,
        enable_async=False,
        attention_bias=True,
        context_format="attention",
        resonance_kernel="cosine",
        phase_coupling=0.0,
        adaptive_bandwidth=False,
        # SOT settings
        sot_enabled=True,
        sot_use_for_query=True,
        sot_subword_seed=True,
        sot_attention_pooling=True,
        sot_hard_negatives=True,
        sot_retrieval_feedback=True,
        sot_max_vocab=5000,
    )

    memory = RTMDKMemory(config=cfg, embedder=dummy_embed)

    # Bootstrap SOT from corpus
    print("Bootstrapping SOT from corpus (100 texts)...")
    texts = [r["query"] + " " + r["answer"] for r in data]
    t0 = time.time()
    memory.field.sot_bootstrap(texts, teacher_model="all-MiniLM-L6-v2")
    print(f"Bootstrap done in {time.time()-t0:.1f}s")

    # Add nodes with dummy embeddings (SOT will learn from co-occurrence)
    for rec in data:
        emb = dummy_embed(rec["query"] + " " + rec["answer"])
        memory.add_node(emb, {"text": rec["answer"]})

    # Evaluate using SOT query_by_text
    hits = 0
    latencies = []
    for rec in data:
        t0 = time.perf_counter()
        result = memory.field.query_by_text(rec["query"], top_k=1)
        latencies.append((time.perf_counter() - t0) * 1000)
        if result:
            top_ctx = result[0][2].content.get("text", "")
            if rec["answer"] in top_ctx:
                hits += 1

    latencies.sort()
    n = len(latencies)
    print(f"\nSOT Benchmark Results (N={len(data)}):")
    print(f"  R@1: {hits}%")
    print(f"  P50 latency: {latencies[n//2]:.2f}ms")
    print(f"  P95 latency: {latencies[int(n*0.95)]:.2f}ms")


if __name__ == "__main__":
    run_sot_benchmark()
