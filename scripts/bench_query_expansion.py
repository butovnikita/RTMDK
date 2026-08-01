"""Benchmark: SOT v2 with PMI-based query expansion."""

import json
import numpy as np


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data["records"] if r.get("language") == "en"]


def evaluate(embedder, corpus_texts, records, top_k=5, expand=False):
    doc_embs = np.vstack([embedder(t) for t in corpus_texts])
    recalls = {1: [], 3: [], 5: []}
    ranks = []
    for rec in records:
        q_emb = embedder(rec["query"], expand_query=expand)
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

    from rtmdk.memory.sot_v2.integration import SOTv2Embedder

    sot = SOTv2Embedder(latent_dim=384, a=0.01, window_size=5)
    sot.train(corpus_texts)

    # Show some expansion examples
    print("\nQuery expansion examples:")
    for rec in records[:3]:
        terms = sot.expand_query_terms(rec["query"], n_terms=3)
        print(f"  Query: {rec['query']}")
        print(f"    Expanded: {terms}")

    baseline_recalls, baseline_mrr = evaluate(sot, corpus_texts, records, expand=False)
    expanded_recalls, expanded_mrr = evaluate(sot, corpus_texts, records, expand=True)

    print(f"\nBaseline:  {baseline_recalls}  MRR={baseline_mrr:.3f}")
    print(f"Expanded:  {expanded_recalls}  MRR={expanded_mrr:.3f}")


if __name__ == "__main__":
    main()
