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

# Trace full field.query
orig_query = field.query

def traced_query(self, embedding, phase=0.0, top_k=None, modality='text', session_id=None, query_text=None):
    total_t0 = time.perf_counter()
    top_k = top_k or self.cfg.top_k
    query_latent = self._project(embedding)
    self._ensure_adaptive_pc(query_latent)
    
    # Query cache
    t0 = time.perf_counter()
    if self.query_cache is not None:
        cache_key = self._query_cache_key(query_latent, phase, top_k, modality, session_id)
        cached = self.query_cache.get_raw(cache_key)
        if cached is not None:
            return cached
    t1 = time.perf_counter()
    print(f'  cache check: {(t1-t0)*1000:.2f}ms')
    
    # HNSW path
    n_pos = len(self.hnsw_index.positions)
    hnsw_k = min(n_pos, max(top_k * 20, min(n_pos // 20, 2000)))
    
    t0 = time.perf_counter()
    candidate_ids = self.hnsw_index.search(query_latent, hnsw_k)
    t1 = time.perf_counter()
    print(f'  hnsw.search: {(t1-t0)*1000:.2f}ms')
    
    candidate_ids = [nid for nid in candidate_ids if nid in self.nodes]
    
    t0 = time.perf_counter()
    scores = self._batch_resonance_cached(
        query_latent[np.newaxis, :],
        np.array([phase], dtype=np.float32),
        candidate_ids,
    )[0]
    t1 = time.perf_counter()
    print(f'  _batch_resonance_cached: {(t1-t0)*1000:.2f}ms')
    
    # Vectorized filter
    t0 = time.perf_counter()
    above = scores >= self.cfg.min_response
    indices = np.where(above)[0]
    results = []
    if len(indices) > 0:
        filtered_scores = scores[indices]
        filtered_ids = [candidate_ids[i] for i in indices]
        n_results = min(len(filtered_scores), top_k)
        if len(filtered_scores) > top_k * 2:
            partition_idx = np.argpartition(filtered_scores, -n_results)[-n_results:]
            top_local = partition_idx[np.argsort(filtered_scores[partition_idx])[::-1]]
        else:
            top_local = np.argsort(filtered_scores)[::-1]
        top_local = top_local[:top_k]
        for ti in top_local:
            nid = filtered_ids[ti]
            node = self.nodes[nid]
            node.last_resonated = time.time()
            results.append((nid, float(filtered_scores[ti]), node))
    t1 = time.perf_counter()
    print(f'  filter+sort+build: {(t1-t0)*1000:.2f}ms')
    
    t0 = time.perf_counter()
    results = self._apply_conformal_filter(results)
    t1 = time.perf_counter()
    print(f'  _apply_conformal_filter: {(t1-t0)*1000:.2f}ms')
    
    t0 = time.perf_counter()
    if self.cfg.adaptive_top_k:
        results = self._apply_adaptive_top_k(results)
    t1 = time.perf_counter()
    print(f'  _apply_adaptive_top_k: {(t1-t0)*1000:.2f}ms')
    
    total = (time.perf_counter() - total_t0) * 1000
    print(f'  TOTAL traced: {total:.2f}ms')
    return results[:top_k]

field.query = types.MethodType(traced_query, field)
print("Query profiling:")
field.query(emb, top_k=5, query_text="document about topic 5")
