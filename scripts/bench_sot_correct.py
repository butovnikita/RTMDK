"""
CORRECT SOT benchmark: both nodes AND queries use SOT embeddings.

Previous test_sot_benchmark.py was broken because:
- Nodes were added with random dummy embeddings
- Queries used SOT embeddings
- Different embedding spaces => ~random results (9% R@1)
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
import numpy as np
import os
import sys
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"


def run_correct_sot_benchmark(n_records=300):
    with open("datasets/qa_1000_en.json", "r", encoding="utf-8") as f:
        data = json.load(f)["records"][:n_records]

    cfg = RTMDKConfig(
        latent_dim=384,  # SBERT dim
        top_k=5,
        min_response=0.001,
        decay_rate=0.999,
        use_hnsw=False,
        learn_projection=False,
        bm25_fallback=False,
        enable_async=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        adaptive_bandwidth=False,
        # SOT settings
        sot_enabled=True,
        sot_use_for_query=True,
        sot_subword_seed=True,
        sot_attention_pooling=True,
        sot_max_vocab=5000,
        sot_bootstrap_corpus="datasets/qa_1000_en.json",
    )

    field = RTMDKField(cfg)

    # Bootstrap SOT from corpus using SBERT teacher
    print("Bootstrapping SOT from corpus...")
    from sentence_transformers import SentenceTransformer
    teacher = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [r["query"] + " " + r["answer"] for r in data]
    field.sot_bootstrap(texts, teacher_model="all-MiniLM-L6-v2")

    # Add nodes using SOT embeddings (NOT external embedder)
    print("Adding nodes with SOT embeddings...")
    for rec in data:
        text = rec["query"] + " " + rec["answer"]
        tokens = field._projection_mgr.sot_tokenizer.encode(text)
        emb = field._projection_mgr.sot_tokenizer.embed(tokens)
        field.add_node(
            emb.astype(np.float32),
            content={"text": rec["answer"]},
            phase=0.0,
            node_id=f"n{hash(text) & 0x7FFFFFFF}",
            skip_projection=True,
        )
        field.nodes[field.node_index[-1]].amplitude = 1.0
        field.nodes[field.node_index[-1]].salience = 1.0

    # Evaluate using SOT query_by_text
    print("Evaluating SOT retrieval...")
    hits = 0
    latencies = []
    for rec in data:
        t0 = time.perf_counter()
        result = field.query_by_text(rec["query"], top_k=1)
        latencies.append((time.perf_counter() - t0) * 1000)
        if result:
            top_ctx = result[0][2].content.get("text", "")
            if rec["answer"] in top_ctx:
                hits += 1

    latencies.sort()
    n = len(latencies)
    print(f"\nCorrect SOT Benchmark (N={len(data)}):")
    print(f"  R@1: {hits / len(data):.1%}")
    print(f"  P50 latency: {latencies[n//2]:.2f}ms")
    print(f"  P95 latency: {latencies[int(n*0.95)]:.2f}ms")

    # Also compare with external SBERT baseline
    print("\n--- SBERT baseline (same data, external embedder) ---")
    cfg2 = RTMDKConfig(
        latent_dim=384, top_k=5, min_response=0.001,
        decay_rate=0.999, use_hnsw=False,
        learn_projection=False, bm25_fallback=False,
        enable_async=False, resonance_kernel="cosine",
        phase_coupling=0.0, adaptive_bandwidth=False,
    )
    field2 = RTMDKField(cfg2)
    for rec in data:
        text = rec["query"] + " " + rec["answer"]
        emb = teacher.encode(text, convert_to_numpy=True).astype(np.float32)
        field2.add_node(
            emb,
            content={"text": rec["answer"]},
            phase=0.0,
            node_id=f"n{hash(text) & 0x7FFFFFFF}",
            skip_projection=True,
        )
        field2.nodes[field2.node_index[-1]].amplitude = 1.0
        field2.nodes[field2.node_index[-1]].salience = 1.0

    hits2 = 0
    for rec in data:
        q_emb = teacher.encode(
            rec["query"],
            convert_to_numpy=True).astype(
            np.float32)
        result = field2.query(q_emb, top_k=1)
        if result:
            top_ctx = result[0][2].content.get("text", "")
            if rec["answer"] in top_ctx:
                hits2 += 1

    print(f"  R@1: {hits2 / len(data):.1%}")


if __name__ == "__main__":
    run_correct_sot_benchmark(300)
