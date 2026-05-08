"""Benchmark: SOT v2.0 baseline vs Procrustes-aligned vs SBERT teacher.

Measures recall@K and MRR on comprehensive_500 (hard paraphrase dataset).
Procrustes alignment distills SBERT knowledge into SIF space without
PyTorch at inference time.
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    # Keep only English for consistency
    records = [r for r in records if r.get("language") == "en"]
    return records


def evaluate(embedder, corpus_texts, records, top_k=5):
    """Evaluate an embedder on a QA dataset."""
    # Encode corpus
    t0 = time.time()
    doc_embs = np.vstack([embedder(t) for t in corpus_texts])
    encode_time = time.time() - t0

    # Encode queries and search
    recalls = []
    ranks = []
    query_times = []

    for rec in records:
        q = rec["query"]
        target = rec["context"]
        target_idx = corpus_texts.index(target)

        t0 = time.time()
        q_emb = embedder(q)
        # Cosine similarity
        sims = doc_embs @ q_emb
        top_indices = np.argsort(-sims)[:top_k]
        query_times.append(time.time() - t0)

        hit = target_idx in top_indices
        recalls.append(1.0 if hit else 0.0)

        # MRR: rank of target (1-indexed)
        sorted_indices = np.argsort(-sims)
        rank = np.where(sorted_indices == target_idx)[0]
        if len(rank) > 0:
            ranks.append(1.0 / (rank[0] + 1))
        else:
            ranks.append(0.0)

    return {
        f"recall@{top_k}": float(np.mean(recalls)),
        "mrr": float(np.mean(ranks)),
        "encode_time_s": encode_time,
        "query_p50_ms": float(np.percentile(query_times, 50) * 1000),
        "query_p99_ms": float(np.percentile(query_times, 99) * 1000),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/comprehensive_500.json")
    parser.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--latent_dim", type=int, default=384)
    parser.add_argument("--a", type=float, default=0.01)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--save_aligner", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    records = load_dataset(args.dataset)
    corpus_texts = list({r["context"] for r in records})
    print(f"Loaded {len(records)} records, {len(corpus_texts)} unique contexts")

    # ---------- 1. SOT v2 baseline ----------
    from rtmdk.memory.sot_v2.integration import SOTv2Embedder

    sot = SOTv2Embedder(latent_dim=args.latent_dim, a=args.a, window_size=args.window)
    sot.train(corpus_texts)
    baseline = evaluate(sot, corpus_texts, records, top_k=args.top_k)
    print(f"\n[SOT v2 baseline]     recall@{args.top_k}={baseline[f'recall@{args.top_k}']:.3f}  MRR={baseline['mrr']:.3f}  query_p50={baseline['query_p50_ms']:.2f}ms")

    # ---------- 2. SBERT teacher ----------
    from sentence_transformers import SentenceTransformer

    teacher = SentenceTransformer(args.teacher)

    class TeacherWrapper:
        def __init__(self, model):
            self.model = model
        def __call__(self, text):
            if isinstance(text, str):
                return self.model.encode(text)
            return self.model.encode(text)

    tw = TeacherWrapper(teacher)
    teacher_res = evaluate(tw, corpus_texts, records, top_k=args.top_k)
    print(f"[SBERT teacher]       recall@{args.top_k}={teacher_res[f'recall@{args.top_k}']:.3f}  MRR={teacher_res['mrr']:.3f}  query_p50={teacher_res['query_p50_ms']:.2f}ms")

    # ---------- 3. Procrustes aligned ----------
    sot_aligned = SOTv2Embedder(latent_dim=args.latent_dim, a=args.a, window_size=args.window)
    sot_aligned.train(corpus_texts)
    sot_aligned.align_to_teacher(corpus_texts, teacher.encode, batch_size=64, center=True)
    aligned = evaluate(sot_aligned, corpus_texts, records, top_k=args.top_k)
    print(f"[SOT v2 + Procrustes] recall@{args.top_k}={aligned[f'recall@{args.top_k}']:.3f}  MRR={aligned['mrr']:.3f}  query_p50={aligned['query_p50_ms']:.2f}ms")

    diag = sot_aligned._embedder._aligner.diagnostics()
    print(f"  Alignment diag: MSE={diag['mse']:.6f}  mean_cosine={diag['mean_cosine']:.4f}")

    if args.save_aligner:
        sot_aligned.save_aligner(args.save_aligner)
        print(f"  Aligner saved to {args.save_aligner}")

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<22}  Recall@{args.top_k}  MRR     Query p50")
    print("-" * 60)
    print(f"{'SOT v2 baseline':<22}  {baseline[f'recall@{args.top_k}']:.3f}       {baseline['mrr']:.3f}   {baseline['query_p50_ms']:.2f}ms")
    print(f"{'SBERT teacher':<22}  {teacher_res[f'recall@{args.top_k}']:.3f}       {teacher_res['mrr']:.3f}   {teacher_res['query_p50_ms']:.2f}ms")
    print(f"{'SOT v2 + Procrustes':<22}  {aligned[f'recall@{args.top_k}']:.3f}       {aligned['mrr']:.3f}   {aligned['query_p50_ms']:.2f}ms")
    print("-" * 60)

    baseline_r = baseline[f"recall@{args.top_k}"]
    aligned_r = aligned[f"recall@{args.top_k}"]
    teacher_r = teacher_res[f"recall@{args.top_k}"]
    gap_closed = (aligned_r - baseline_r) / max(teacher_r - baseline_r, 1e-8) * 100
    print(f"Gap closed vs SBERT: {gap_closed:.1f}%")


if __name__ == "__main__":
    main()
