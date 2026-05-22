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
print(f"HNSW enabled: {cfg.use_hnsw}")
print(f"HNSW index type: {type(field.hnsw_index).__name__}")
print(f"HNSW positions: {len(field.hnsw_index.positions) if field.hnsw_index else 0}")

emb = embed("document about topic 5")

# Trace query
t0 = time.perf_counter()
results = field.query(emb, top_k=5, query_text="document about topic 5")
lat = (time.perf_counter() - t0) * 1000
print(f"field.query latency: {lat:.2f}ms, results={len(results)}")

# Direct HNSW search
if field.hnsw_index:
    t0 = time.perf_counter()
    cands = field.hnsw_index.search(field._project(emb), 100)
    hnsw_lat = (time.perf_counter() - t0) * 1000
    print(f"HNSW search latency: {hnsw_lat:.2f}ms, candidates={len(cands)}")

# _query_vectorized latency
t0 = time.perf_counter()
results2 = field._query_vectorized(field._project(emb), 0.0, 5, "text", None, 0)
vec_lat = (time.perf_counter() - t0) * 1000
print(f"_query_vectorized latency: {vec_lat:.2f}ms, results={len(results2)}")
