import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rtmdk.memory.sot_v2.integration import SOTv2Embedder, _word_tokenize

with open("datasets/comprehensive_500.json", "r", encoding="utf-8") as f:
    records = json.load(f)["records"]

records = [r for r in records if r.get("language") == "en"]
corpus_texts = list({r["context"] for r in records})
queries = [r["query"] for r in records]

# Teacher
teacher = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# SOT v2
sot = SOTv2Embedder(latent_dim=384, window_size=5, a=0.01, remove_pc=True)
all_texts = list(corpus_texts) + queries
sot.train(all_texts)
sot.align_to_teacher(all_texts, teacher.encode, batch_size=64, center=True)

# Check dimensions
q_emb_sot = sot(queries[0])
q_emb_teacher = teacher.encode(queries[0])
print(f"SOT dim: {q_emb_sot.shape}, Teacher dim: {q_emb_teacher.shape}")

# Check a few query-context similarities
target_ctx = records[0]["context"]
ctx_idx = corpus_texts.index(target_ctx)

for i in range(5):
    q = queries[i]
    target = records[i]["context"]
    
    q_sot = sot(q)
    c_sot = sot(target)
    q_t = teacher.encode(q)
    c_t = teacher.encode(target)
    
    sim_sot = float(np.dot(q_sot, c_sot))
    sim_t = float(np.dot(q_t, c_t))
    
    print(f"Q{i}: {q[:50]}")
    print(f"  SOT cos: {sim_sot:.3f}  Teacher cos: {sim_t:.3f}")
    
    # Rank target among all contexts
    ctx_sims_sot = [float(np.dot(q_sot, sot(c))) for c in corpus_texts]
    ctx_sims_t = [float(np.dot(q_t, teacher.encode(c))) for c in corpus_texts]
    
    rank_sot = sorted(ctx_sims_sot, reverse=True).index(ctx_sims_sot[corpus_texts.index(target)]) + 1
    rank_t = sorted(ctx_sims_t, reverse=True).index(ctx_sims_t[corpus_texts.index(target)]) + 1
    
    print(f"  SOT rank: {rank_sot}/{len(corpus_texts)}  Teacher rank: {rank_t}/{len(corpus_texts)}")
