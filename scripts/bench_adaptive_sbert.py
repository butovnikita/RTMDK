"""
A/B benchmark: adaptive_bandwidth on REAL semantic embeddings (SBERT).
Uses sentence-transformers (no LM Studio needed).
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rtmdk.memory.field import RTMDKField
from rtmdk.memory.config import RTMDKConfig

os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"
np.random.seed(42)


def load_dataset(path="datasets/qa_1000_en.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


def get_sbert():
    from sentence_transformers import SentenceTransformer
    print("Loading SBERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")
    return model


def build_field(records, cfg, model, skip_proj=True):
    field = RTMDKField(cfg)
    texts = [r["context"] for r in records]
    print(f"  Encoding {len(texts)} contexts...")
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for rec, emb in zip(records, embs):
        field.add_node(
            emb.astype(np.float32),
            content={"text": rec["context"], "answer": rec["answer"]},
            phase=0.0,
            node_id=rec.get("id") or f"n{hash(rec['context']) & 0x7FFFFFFF}",
            skip_projection=skip_proj,
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
        if top_text == rec["context"]:
            correct_1 += 1
        found = any(r[2].content.get("text") == rec["context"] for r in results)
        if found:
            correct_k += 1
        total += 1
    return {"R@1": correct_1 / total, "R@5": correct_k / total, "n": total}


def run(name, cfg, records, model, skip_proj=True):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    field = build_field(records, cfg, model, skip_proj=skip_proj)
    build_t = time.time() - t0
    stats = evaluate(field, records, model, top_k=5)
    print(f"  Build: {build_t:.1f}s  R@1: {stats['R@1']:.3f}  R@5: {stats['R@5']:.3f}")
    return field, stats


def main():
    records = load_dataset()
    subset = records[:300]  # SBERT encoding is slower; use 300 for speed
    print(f"Dataset: {len(records)} records, using subset: {len(subset)}")

    model = get_sbert()

    # Note: SBERT dim=384. We can either use latent_dim=384 (no projection)
    # or latent_dim=128 (with projection) to match production config.

    # Test 1: No projection, global bw
    cfg1 = RTMDKConfig(
        latent_dim=384, bandwidth=1.0, adaptive_bandwidth=False,
        resonance_kernel="cosine", phase_coupling=0.0,
        min_response=0.001, use_hnsw=False,
    )
    f1, s1 = run("No proj, global bw", cfg1, subset, model, skip_proj=True)

    # Test 2: No projection, adaptive k=5
    cfg2 = RTMDKConfig(
        latent_dim=384, bandwidth=1.0, adaptive_bandwidth=True,
        adaptive_bandwidth_k=5, adaptive_bandwidth_min_n=50,
        resonance_kernel="cosine", phase_coupling=0.0,
        min_response=0.001, use_hnsw=False,
    )
    f2, s2 = run("No proj, adaptive k=5", cfg2, subset, model, skip_proj=True)

    # Test 3: Projection 384->128, global bw
    cfg3 = RTMDKConfig(
        latent_dim=128, bandwidth=1.0, adaptive_bandwidth=False,
        resonance_kernel="cosine", phase_coupling=0.0,
        min_response=0.001, use_hnsw=False, learn_projection=False,
    )
    f3, s3 = run("Proj 384->128, global bw", cfg3, subset, model, skip_proj=False)

    # Test 4: Projection 384->128, adaptive k=5
    cfg4 = RTMDKConfig(
        latent_dim=128, bandwidth=1.0, adaptive_bandwidth=True,
        adaptive_bandwidth_k=5, adaptive_bandwidth_min_n=50,
        resonance_kernel="cosine", phase_coupling=0.0,
        min_response=0.001, use_hnsw=False, learn_projection=False,
    )
    f4, s4 = run("Proj 384->128, adaptive k=5", cfg4, subset, model, skip_proj=False)

    # Test 5: Projection 384->128, adaptive k=20 (more stable density est)
    cfg5 = RTMDKConfig(
        latent_dim=128, bandwidth=1.0, adaptive_bandwidth=True,
        adaptive_bandwidth_k=20, adaptive_bandwidth_min_n=50,
        resonance_kernel="cosine", phase_coupling=0.0,
        min_response=0.001, use_hnsw=False, learn_projection=False,
    )
    f5, s5 = run("Proj 384->128, adaptive k=20", cfg5, subset, model, skip_proj=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    rows = [
        ("No proj, global", s1),
        ("No proj, adaptive k=5", s2),
        ("Proj 384->128, global", s3),
        ("Proj 384->128, adaptive k=5", s4),
        ("Proj 384->128, adaptive k=20", s5),
    ]
    for name, st in rows:
        print(f"  {name:35s}  R@1={st['R@1']:.3f}  R@5={st['R@5']:.3f}")


if __name__ == "__main__":
    main()
