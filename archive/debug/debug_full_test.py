"""debug_full_test.py"""
import numpy as np
from embedder_lmstudio import LMStudioEmbedder
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory

emb = LMStudioEmbedder()

# Replicate the exact benchmark scenario
facts = []
for i in range(100):
    topic = ["science", "history", "geography", "tech", "health"][i % 5]
    kw = f"fc_kw_{i:05d}"
    facts.append({
        "fact": f"{topic} fact number {i} with unique keyword {kw}",
        "query": f"What is the {topic} fact number {i} keyword?",
        "keyword": kw,
    })

config = RTMDKConfig(
    embedding_dim=768, latent_dim=128, top_k=5, min_response=0.001,
    decay_rate=0.999, enable_async=False, bm25_fallback=True,
    use_hnsw=False, learn_projection=False,
)
mem = RTMDKMemory(config=config, embedder=emb)

# Store all facts
for item in facts:
    mem.save_context({"input": item["fact"], "session_id": "fc"}, {"output": item["fact"]})
    mem.save_context({"input": item["query"], "session_id": "fc"}, {"output": item["fact"]})
    mem.save_context({"input": item["keyword"], "session_id": "fc"}, {"output": item["fact"]})

print(f"Total nodes: {len(mem.field.nodes)}")

# Test recall on first 50
test_facts = facts[:50]
n_correct = 0
for item in test_facts:
    ctx = mem.load_memory_variables({"input": item["query"], "session_id": "fc"})
    c = ctx["rtmdk_context"].lower()
    found = item["keyword"] in c
    if not found:
        # Debug first miss
        if n_correct < 3:
            print(f"  MISS: keyword={item['keyword']}")
            print(f"    query: {item['query'][:60]}")
            print(f"    context[:150]: {ctx['rtmdk_context'][:150]}")
    if found:
        n_correct += 1

recall = n_correct / len(test_facts)
print(f"\nRecall: {n_correct}/{len(test_facts)} = {recall:.2%}")
print(f"Nodes: {len(mem.field.nodes)}")
