"""Detailed recall@K analysis for SOT v2 vs SBERT on comprehensive_500."""

import json
import numpy as np


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data["records"] if r.get("language") == "en"]


def evaluate(embedder, corpus_texts, records):
    doc_embs = np.vstack([embedder(t) for t in corpus_texts])
    recalls = {1: [], 3: [], 5: [], 10: []}
    ranks = []
    for rec in records:
        q_emb = embedder(rec["query"])
        sims = doc_embs @ q_emb
        sorted_idx = np.argsort(-sims)
        target_idx = corpus_texts.index(rec["context"])
        rank = np.where(sorted_idx == target_idx)[0][0] + 1
        ranks.append(1.0 / rank)
        for k in recalls:
            recalls[k].append(1.0 if target_idx in sorted_idx[:k] else 0.0)
    return {f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()}, float(np.mean(ranks))


def main():
    records = load_dataset("datasets/comprehensive_500.json")
    corpus_texts = list({r["context"] for r in records})
    print(f"Records: {len(records)}, contexts: {len(corpus_texts)}")

    # SOT v2
    from rtmdk.memory.sot_v2.integration import SOTv2Embedder
    sot = SOTv2Embedder(latent_dim=384, a=0.01, window_size=5)
    sot.train(corpus_texts)
    sot_recalls, sot_mrr = evaluate(sot, corpus_texts, records)
    print(f"SOT v2:    {sot_recalls}  MRR={sot_mrr:.3f}")

    # SBERT
    from sentence_transformers import SentenceTransformer
    teacher = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    t_recalls, t_mrr = evaluate(lambda t: teacher.encode(t), corpus_texts, records)
    print(f"SBERT:     {t_recalls}  MRR={t_mrr:.3f}")

    # SOT v2 with a=0.1, window=20 (bad config from grid search)
    sot_bad = SOTv2Embedder(latent_dim=384, a=0.1, window_size=20)
    sot_bad.train(corpus_texts)
    bad_recalls, bad_mrr = evaluate(sot_bad, corpus_texts, records)
    print(f"SOT v2 bad:{bad_recalls}  MRR={bad_mrr:.3f}")


if __name__ == "__main__":
    main()
