import os
os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

import time
import numpy as np
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory

cfg = RTMDKConfig(
    latent_dim=384, embedding_dim=384, max_nodes=5100, top_k=5,
    min_response=0.001, bandwidth=1.0, phase_coupling=0.0, use_hnsw=False,
    pipeline_enabled=True,
)

def embed(text):
    h = hash(text) % (2**32)
    r = np.random.default_rng(h)
    return r.standard_normal(384).astype(np.float32)

memory = RTMDKMemory(config=cfg, embedder=embed)
for i in range(5000):
    text = f"document about topic {i % 100} aspect {i} keywords {' '.join(__import__('random').choices(['neural', 'embedding', 'search', 'vector', 'memory'], k=5))}"
    memory.add_node(content={"text": text}, embedding=embed(text))

q = "document about topic 5"

# Warmup
for _ in range(3):
    memory.retrieve_nodes_pipeline(q, top_k=5)

latencies = []
for _ in range(20):
    t0 = time.perf_counter()
    memory.retrieve_nodes_pipeline(q, top_k=5)
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"p50={np.percentile(latencies,50):.2f}ms p95={np.percentile(latencies,95):.2f}ms p99={np.percentile(latencies,99):.2f}ms")

# Direct field.query latency
field = memory.field
emb = embed(q)
t0 = time.perf_counter()
field.query(emb, top_k=5, query_text=q)
print(f"direct field.query (with query_text): {(time.perf_counter()-t0)*1000:.2f}ms")
