"""Diagnose: why does RTMDK pipeline degrade SOT v2 recall?"""

import json
import numpy as np


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data["records"] if r.get("language") == "en"]


def main():
    records = load_dataset("datasets/comprehensive_500.json")
    corpus_texts = list({r["context"] for r in records})
    print(f"Records: {len(records)}, Unique contexts: {len(corpus_texts)}")

    # 1. Pure cosine on SOT v2 embeddings
    from rtmdk.memory.sot_v2.integration import SOTv2Embedder

    sot = SOTv2Embedder(latent_dim=384, a=0.01, window_size=5)
    sot.train(corpus_texts)
    doc_embs = np.vstack([sot(t) for t in corpus_texts])

    def pure_cosine_recall(top_k=5):
        hits = 0
        for rec in records:
            q_emb = sot(rec["query"])
            sims = doc_embs @ q_emb
            top_idx = np.argsort(-sims)[:top_k]
            target_idx = corpus_texts.index(rec["context"])
            if target_idx in top_idx:
                hits += 1
        return hits / len(records)

    print(f"Pure cosine recall@5: {pure_cosine_recall(5):.3f}")

    # 2. Through RTMDKMemory.retrieve_nodes
    from rtmdk.memory.core import RTMDKMemory
    from rtmdk.memory.config import RTMDKConfig, SOTConfig, CoreConfig, RetrievalConfig, RoutingConfig

    cfg = RTMDKConfig(
        core=CoreConfig(latent_dim=384, top_k=5, min_response=0.0),
        retrieval=RetrievalConfig(use_hnsw=True),  # enable HNSW
        routing=RoutingConfig(sparse_routing=False),
        sot=SOTConfig(
            sot_v2_enabled=True,
            sot_v2_a=0.01,
            sot_v2_window=5,
            sot_v2_remove_pc=True,
        ),
    )
    mem = RTMDKMemory(config=cfg, embedder=sot)
    mem.train_sot_v2()

    # Add contexts as nodes
    for ctx in corpus_texts:
        emb = sot(ctx)
        mem.add_node(emb, {"text": ctx})

    # Query via retrieve_nodes
    hits = 0
    for rec in records:
        q_emb = sot(rec["query"])
        results = mem.retrieve_nodes(rec["query"], q_emb, top_k=5)
        contexts = [r[2].content.get("text", "") for r in results]
        if rec["context"] in contexts:
            hits += 1

    print(f"RTMDK retrieve_nodes recall@5: {hits / len(records):.3f}")

    # 3. Through field.query directly
    hits2 = 0
    for rec in records:
        emb = sot(rec["query"])
        phase = mem._get_phase(None, emb)
        results = mem.field.query(emb, phase, top_k=5)
        contexts = [r[2].content.get("text", "") for r in results]
        if rec["context"] in contexts:
            hits2 += 1

    print(f"field.query recall@5: {hits2 / len(records):.3f}")


if __name__ == "__main__":
    main()
