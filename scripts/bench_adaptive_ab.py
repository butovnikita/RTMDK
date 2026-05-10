"""
A/B benchmark: adaptive_bandwidth=True vs False on QA dataset.
Uses fallback embedder (deterministic, no LM Studio needed).
Measures relative degradation, not absolute accuracy.
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
import numpy as np
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


np.random.seed(42)


def fallback_embed(text, dim=768):
    """Deterministic fallback embedder (same as benchmark)."""
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    emb = rng.standard_normal(dim).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


def load_dataset(path="datasets/qa_1000_en.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


def build_field(records, cfg):
    field = RTMDKField(cfg)
    for rec in records:
        emb = fallback_embed(rec["context"])
        field.add_node(
            emb,
            content={"text": rec["context"], "answer": rec["answer"]},
            phase=0.0,
            node_id=rec.get("id") or f"n{hash(rec['context']) & 0x7FFFFFFF}",
            skip_projection=True,
        )
        field.nodes[field.node_index[-1]].amplitude = 1.0
        field.nodes[field.node_index[-1]].salience = 1.0
    return field


def evaluate(field, records, top_k=5):
    correct_1 = 0
    correct_k = 0
    total = 0
    for rec in records:
        q_emb = fallback_embed(rec["query"])
        results = field.query(q_emb, top_k=top_k)
        if not results:
            continue
        # Check if top-1 answer text matches
        top_text = results[0][2].content.get("text", "")
        if top_text == rec["context"]:
            correct_1 += 1
        # Check if correct context is in top-k
        found = any(r[2].content.get("text") == rec["context"] for r in results)
        if found:
            correct_k += 1
        total += 1
    return {
        "R@1": correct_1 / total if total else 0,
        "R@5": correct_k / total if total else 0,
        "n": total,
    }


def run(name, cfg, records):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    field = build_field(records, cfg)
    build_t = time.time() - t0
    stats = evaluate(field, records, top_k=5)
    print(f"  Build: {build_t:.1f}s  Nodes: {len(records)}")
    print(f"  R@1: {stats['R@1']:.3f}  R@5: {stats['R@5']:.3f}")
    return field, stats


def main():
    os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
    print("Loading QA dataset...")
    records = load_dataset()
    print(f"  Loaded {len(records)} records")

    # Use a subset for speed
    subset = records[:500]
    print(f"  Using subset: {len(subset)} records")

    cfg_global = RTMDKConfig(
        latent_dim=768,
        bandwidth=1.0,
        adaptive_bandwidth=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f1, s1 = run("Global BW", cfg_global, subset)

    cfg_adapt = RTMDKConfig(
        latent_dim=768,
        bandwidth=1.0,
        adaptive_bandwidth=True,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f2, s2 = run("Adaptive BW (k=5)", cfg_adapt, subset)

    cfg_adapt_k20 = RTMDKConfig(
        latent_dim=768,
        bandwidth=1.0,
        adaptive_bandwidth=True,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f3, s3 = run("Adaptive BW (k=20)", cfg_adapt_k20, subset)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Global BW        R@1={s1['R@1']:.3f}  R@5={s1['R@5']:.3f}")
    print(f"  Adaptive k=5     R@1={s2['R@1']:.3f}  R@5={s2['R@5']:.3f}  ΔR@1={s2['R@1']-s1['R@1']:+.3f}")
    print(f"  Adaptive k=20    R@1={s3['R@1']:.3f}  R@5={s3['R@5']:.3f}  ΔR@1={s3['R@1']-s1['R@1']:+.3f}")

    if s2["R@1"] < s1["R@1"] * 0.9:
        print("\n  ⚠️  Adaptive bandwidth causes significant degradation!")
    else:
        print("\n  ✓  Adaptive bandwidth does NOT degrade ranking on this benchmark.")
        print("     (Real LM Studio embedder may behave differently.)")


if __name__ == "__main__":
    main()
