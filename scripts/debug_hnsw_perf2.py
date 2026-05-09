import os
os.environ["RTMDK_ADD_RATE_LIMIT"] = "0"

import time
import types
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

# Warmup
field.query(emb, top_k=5, query_text="document about topic 5")

# Trace field.query HNSW path
orig_query = field.query

def traced_query(self, embedding, phase=0.0, top_k=None, modality='text', session_id=None, query_text=None):
    t0 = time.perf_counter()
    top_k = top_k or self.cfg.top_k
    query_latent = self._project(embedding)
    self._ensure_adaptive_pc(query_latent)
    t1 = time.perf_counter()
    print(f'  setup: {(t1-t0)*1000:.2f}ms')
    
    n_pos = len(self.hnsw_index.positions)
    hnsw_k = min(n_pos, max(top_k * 20, min(n_pos // 20, 2000)))
    
    t0 = time.perf_counter()
    candidate_ids = self.hnsw_index.search(query_latent, hnsw_k)
    t1 = time.perf_counter()
    print(f'  hnsw.search: {(t1-t0)*1000:.2f}ms, k={hnsw_k}, cands={len(candidate_ids)}')
    
    t0 = time.perf_counter()
    candidate_ids = [nid for nid in candidate_ids if nid in self.nodes]
    t1 = time.perf_counter()
    print(f'  filter candidates: {(t1-t0)*1000:.2f}ms')
    
    t0 = time.perf_counter()
    scores = self._batch_resonance_cached(
        query_latent[np.newaxis, :],
        np.array([phase], dtype=np.float32),
        candidate_ids,
    )[0]
    t1 = time.perf_counter()
    print(f'  _batch_resonance_cached: {(t1-t0)*1000:.2f}ms')
    
    above = scores >= self.cfg.min_response
    indices = np.where(above)[0]
    print(f'  above threshold: {len(indices)}/{len(scores)}')
    
    t0 = time.perf_counter()
    # Original query continues...
    results = orig_query(embedding, phase, top_k, modality, session_id, query_text)
    t1 = time.perf_counter()
    print(f'  TOTAL orig_query: {(t1-t0)*1000:.2f}ms')
    return results

field.query = types.MethodType(traced_query, field)
print("Query profiling:")
field.query(emb, top_k=5, query_text="document about topic 5")
