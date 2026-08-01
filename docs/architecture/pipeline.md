# Pipeline Architecture (v8.3+)

RTMDK v8.3 replaces the monolithic `retrieve_nodes()` with an **explicit
6-stage retrieval pipeline** — independently observable, swappable, and
protected by per-stage circuit breakers. The legacy API remains
backward-compatible.

```
Embed → Route → Retrieve → Rerank → Calibrate → Explain
```

| # | Stage | Purpose | Config Flag |
|---|-------|---------|-------------|
| 1 | **Embed** | Query text → embedding | — |
| 2 | **Route** | Query classification (fast / standard / deep) | `cascade_enabled` |
| 3 | **Retrieve** | Resonance / HNSW / BM25 hybrid | — |
| 4 | **Rerank** | Sentence-level + cross-encoder reranking | `sentence_reranker_enabled` |
| 5 | **Calibrate** | Conformal prediction filtering | `conformal_prediction` |
| 6 | **Explain** | Human-readable result explanations | `result_explainability_enabled` |

## Usage

```python
config = RTMDKConfig(pipeline_enabled=True)  # legacy calls use pipeline internally
memory = RTMDKMemory(config=config, embedder=embed_fn)

# Or call the pipeline API directly for full observability:
output = memory.retrieve_nodes_pipeline("What is resonance?", top_k=5)
# output["results"]      — ranked nodes
# output["route"]        — "fast" | "standard" | "deep"
# output["explanations"] — per-result reasons
# output["metrics"]      — per-stage latency breakdown
```

## HTTP Endpoints

```bash
curl http://localhost:8080/v1/memory/pipeline/health    # per-stage health + breaker states
curl http://localhost:8080/v1/memory/pipeline/metrics   # aggregated metrics
curl http://localhost:8080/v1/memory/pipeline/plan      # preview execution plan
```

## Full Documentation

- [Pipeline Architecture — stages, metrics format, streaming, custom pipelines](../PIPELINE_ARCHITECTURE.md)
- [Pipeline Migration Guide — migrating from legacy `retrieve_nodes()`](../21_PIPELINE_MIGRATION.md)
- [ADR 001 — design decision and trade-offs](../ADR_001_PIPELINE_V83.md)
