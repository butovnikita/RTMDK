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
    pipeline_enabled=True,
    pipeline_cost_tracking_enabled=True,
)

def embed(text):
    h = hash(text) % (2**32)
    r = np.random.default_rng(h)
    return r.standard_normal(384).astype(np.float32)

memory = RTMDKMemory(config=cfg, embedder=embed)
for i in range(10000):
    text = f"document about topic {i % 100} aspect {i}"
    memory.add_node(content={"text": text}, embedding=embed(text))

q = "document about topic 5"

# Warmup
memory.retrieve_nodes_pipeline(q, top_k=5)

# Test retrieve_nodes_pipeline
print("retrieve_nodes_pipeline:")
for i in range(5):
    t0 = time.perf_counter()
    result = memory.retrieve_nodes_pipeline(q, top_k=5)
    lat = (time.perf_counter() - t0) * 1000
    print(f"  q{i+1}: {lat:.2f}ms")

# Test direct field.query
field = memory.field
emb = embed(q)
print("\nfield.query direct:")
for i in range(5):
    t0 = time.perf_counter()
    field.query(emb, top_k=5, query_text=q)
    lat = (time.perf_counter() - t0) * 1000
    print(f"  q{i+1}: {lat:.2f}ms")
