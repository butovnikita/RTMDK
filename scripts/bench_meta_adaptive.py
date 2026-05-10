"""
Benchmark MetaAdaptiveKernel vs global bandwidth vs P1.2 local adaptive bw.

Tests the ORIGINAL v8 adaptive bandwidth concept (MetaAdaptiveKernel),
not the broken P1.2 k-NN local bandwidth.
"""

from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.field import RTMDKField
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
np.random.seed(42)


def load_data(n=300):
    with open("datasets/qa_1000_en.json", "r", encoding="utf-8") as f:
        return json.load(f)["records"][:n]


def build_field(records, cfg, model):
    field = RTMDKField(cfg)
    texts = [r["query"] + " " + r["answer"] for r in records]
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    for rec, emb in zip(records, embs):
        field.add_node(
            emb.astype(np.float32),
            content={"text": rec["answer"]},
            phase=0.0,
            node_id=f"n{hash(rec['query'] + rec['answer']) & 0x7FFFFFFF}",
            skip_projection=True,
        )
        field.nodes[field.node_index[-1]].amplitude = 1.0
        field.nodes[field.node_index[-1]].salience = 1.0
    return field


def evaluate(field, records, model, top_k=5):
    correct_1 = 0
    correct_k = 0
    total = 0
    for rec in records:
        q_emb = model.encode(rec["query"], convert_to_numpy=True).astype(np.float32)
        results = field.query(q_emb, top_k=top_k)
        if not results:
            continue
        top_text = results[0][2].content.get("text", "")
        if top_text == rec["answer"]:
            correct_1 += 1
        found = any(r[2].content.get("text") == rec["answer"] for r in results)
        if found:
            correct_k += 1
        total += 1
    return {"R@1": correct_1 / total, "R@5": correct_k / total}


def run(name, cfg, records, model):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")
    field = build_field(records, cfg, model)
    stats = evaluate(field, records, model, top_k=5)
    bw_final = field.meta_kernel.get_bandwidth() if field.meta_kernel else cfg.bandwidth
    print(f"  R@1: {stats['R@1']:.3f}  R@5: {stats['R@5']:.3f}  final_bw: {bw_final:.3f}")
    return field, stats


def main():
    records = load_data(n=300)
    print(f"Dataset: {len(records)} records")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1. Global bandwidth (baseline)
    cfg_global = RTMDKConfig(
        latent_dim=384,
        bandwidth=1.0,
        adaptive_bandwidth=False,
        meta_adaptive=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f1, s1 = run("Global BW (baseline)", cfg_global, records, model)

    # 2. MetaAdaptiveKernel
    cfg_meta = RTMDKConfig(
        latent_dim=384,
        bandwidth=1.0,
        adaptive_bandwidth=False,
        meta_adaptive=True,
        meta_adaptation_lr=0.005,
        kurtosis_target_min=1.5,
        kurtosis_target_max=4.0,
        resonance_kernel="cosine",
        phase_coupling=0.3,
        min_response=0.001,
        use_hnsw=False,
    )
    f2, s2 = run("MetaAdaptiveKernel", cfg_meta, records, model)

    # 3. P1.2 local adaptive bw (for comparison — known to be useless)
    cfg_local = RTMDKConfig(
        latent_dim=384,
        bandwidth=1.0,
        adaptive_bandwidth=True,
        meta_adaptive=False,
        resonance_kernel="cosine",
        phase_coupling=0.0,
        min_response=0.001,
        use_hnsw=False,
    )
    f3, s3 = run("P1.2 Local Adaptive BW", cfg_local, records, model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Global BW (baseline)       R@1={s1['R@1']:.3f}")
    print(f"  MetaAdaptiveKernel         R@1={s2['R@1']:.3f}  delta={s2['R@1']-s1['R@1']:+.3f}")
    print(f"  P1.2 Local Adaptive BW     R@1={s3['R@1']:.3f}  delta={s3['R@1']-s1['R@1']:+.3f}")

    if s2["R@1"] >= s1["R@1"] * 0.95:
        print("\n  MetaAdaptiveKernel: SAFE (no significant degradation)")
    else:
        print(f"\n  MetaAdaptiveKernel: DEGRADES by {1-s2['R@1']/s1['R@1']:.1%}")


if __name__ == "__main__":
    main()
