import os
os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

import time
import types
import numpy as np
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory

cfg = RTMDKConfig(
    latent_dim=384, embedding_dim=384, max_nodes=100100, top_k=5,
    min_response=0.001, bandwidth=1.0, phase_coupling=0.0,
    use_hnsw=True, hnsw_min_nodes=10,
)

def embed(text):
    h = hash(text) % (2**32)
    r = np.random.default_rng(h)
    return r.standard_normal(384).astype(np.float32)

memory = RTMDKMemory(config=cfg, embedder=embed)

print("Inserting 100K nodes...")
t0 = time.perf_counter()
for i in range(100000):
    text = f"document about topic {i % 100} aspect {i}"
    memory.add_node(content={"text": text}, embedding=embed(text))
    if i > 0 and i % 10000 == 0:
        print(f"  {i} nodes ({i/(time.perf_counter()-t0):.0f}/sec)")
print(f"Insert done: {time.perf_counter()-t0:.1f}s")

field = memory.field
emb = embed("document about topic 5")

# Monkey-patch query to trace
orig_query = field.query

def traced_query(self, embedding, phase=0.0, top_k=None, modality='text', session_id=None, query_text=None):
    t0 = time.perf_counter()
    top_k = top_k or self.cfg.top_k
    query_latent = self._project(embedding)
    t1 = time.perf_counter()
    print(f'  _project: {(t1-t0)*1000:.2f}ms')
    
    t0 = time.perf_counter()
    self._ensure_adaptive_pc(query_latent)
    t1 = time.perf_counter()
    print(f'  _ensure_adaptive_pc: {(t1-t0)*1000:.2f}ms')
    
    # Check which path is taken
    has_hnsw = self.cfg.use_hnsw and self.hnsw_index and len(self.hnsw_index.positions) > getattr(self.cfg, "hnsw_min_nodes", 50)
    print(f'  HNSW path active: {has_hnsw}, positions={len(self.hnsw_index.positions) if self.hnsw_index else 0}')
    
    t0 = time.perf_counter()
    results = orig_query(embedding, phase, top_k, modality, session_id, query_text)
    t1 = time.perf_counter()
    print(f'  TOTAL query: {(t1-t0)*1000:.2f}ms, results={len(results)}')
    return results

field.query = types.MethodType(traced_query, field)

print("\nQuery profiling:")
field.query(emb, top_k=5, query_text="document about topic 5")
