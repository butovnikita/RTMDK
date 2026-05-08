import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory

with open("datasets/comprehensive_500.json", "r", encoding="utf-8") as f:
    records = json.load(f)["records"]

records = [r for r in records if r.get("language") == "en"]
corpus_texts = list({r["context"] for r in records})

teacher = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder = lambda t: teacher.encode(t)

cfg = RTMDKConfig(latent_dim=384, top_k=5, use_hnsw=False, sparse_routing=False)
mem = RTMDKMemory(config=cfg, embedder=embedder)

for ctx in corpus_texts:
    mem.add_node(embedder(ctx), {"text": ctx})

# Check a few queries
for i in range(10):
    rec = records[i]
    q = rec["query"]
    target = rec["context"]
    q_emb = embedder(q)
    
    results = mem.retrieve_nodes(q, q_emb, top_k=5)
    contexts = [r[2].content.get("text", "") for r in results]
    scores = [r[1] for r in results]
    
    if target in contexts:
        rank = contexts.index(target) + 1
        print(f"Q{i}: rank={rank} score={scores[rank-1]:.3f} | {q[:50]}")
    else:
        print(f"Q{i}: MISSING | {q[:50]}")
        # Find target score manually
        target_idx = corpus_texts.index(target)
        # Compute cosine similarity for reference
        doc_embs = np.vstack([embedder(c) for c in corpus_texts])
        doc_embs = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        q_emb_n = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        cossims = doc_embs @ q_emb_n
        target_rank = np.argsort(-cossims).tolist().index(target_idx) + 1
        print(f"     Cosine rank: {target_rank}")

# Overall stats
hits5 = 0
hits1 = 0
mrr = 0
for rec in records:
    q = rec["query"]
    target = rec["context"]
    results = mem.retrieve_nodes(q, embedder(q), top_k=5)
    contexts = [r[2].content.get("text", "") for r in results]
    if target in contexts:
        hits5 += 1
        rank = contexts.index(target) + 1
        mrr += 1.0 / rank
        if rank == 1:
            hits1 += 1

print(f"\nOverall: recall@1={hits1/len(records):.3f} recall@5={hits5/len(records):.3f} MRR={mrr/len(records):.3f}")
