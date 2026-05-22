import os
os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

import time
import numpy as np
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory

cfg = RTMDKConfig(
    latent_dim=384, embedding_dim=384, max_nodes=10100, top_k=5,
    min_response=0.001, bandwidth=1.0, phase_coupling=0.0,
    use_hnsw=True, hnsw_min_nodes=10,
)

def embed(text):
    h = hash(text) % (2**32)
    r = np.random.default_rng(h)
    return r.standard_normal(384).astype(np.float32)

memory = RTMDKMemory(config=cfg, embedder=embed)
for i in range(10000):
    text = f"document about topic {i % 100} aspect {i}"
    memory.add_node(content={"text": text}, embedding=embed(text))

field = memory.field
emb = embed("document about topic 5")

# First query (may build cache)
print("First query:")
t0 = time.perf_counter()
field.query(emb, top_k=5, query_text="document about topic 5")
print(f"  total: {(time.perf_counter()-t0)*1000:.2f}ms")

# Subsequent queries
print("\nSubsequent queries:")
for i in range(5):
    t0 = time.perf_counter()
    field.query(emb, top_k=5, query_text="document about topic 5")
    print(f"  q{i+1}: {(time.perf_counter()-t0)*1000:.2f}ms")
