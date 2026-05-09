# RTMDK Pipeline Architecture (v8.3+)

> Status: Beta — backward-compatible with legacy `retrieve_nodes()` API.

## Overview

RTMDK v8.3 introduces an **explicit stage-based retrieval pipeline** that replaces the monolithic `retrieve_nodes()` method with independently observable, swappable, and testable components.

## Pipeline Stages

| # | Stage | Class | Purpose | Config Flag |
|---|-------|-------|---------|-------------|
| 1 | **Embed** | `EmbedStage` | Text → embedding vector | — |
| 2 | **Route** | `RouteStage` | Query classification (factual/exploratory/deep) | `cascade_enabled` |
| 3 | **Retrieve** | `RetrieveStage` | Core resonance / HNSW / BM25 hybrid | — |
| 4 | **Rerank** | `RerankStage` | Sentence-level + cross-encoder reranking | `sentence_reranker_enabled` |
| 5 | **Calibrate** | `CalibrateStage` | Conformal prediction filtering | `conformal_prediction` |
| 6 | **Explain** | `ExplainStage` | Human-readable result explanations | `result_explainability_enabled` |

## Legacy vs Pipeline API

### Legacy (monolithic)
```python
results = memory.retrieve_nodes(query, embedding, top_k=5)
```

### Pipeline (observable)
```python
output = memory.retrieve_nodes_pipeline(query, embedding, top_k=5)
# output["results"]      — list of (nid, score, node)
# output["route"]        — "fast" | "standard" | "deep"
# output["explanations"] — human-readable reasons
# output["metrics"]      — per-stage latency breakdown
```

### Custom Pipeline
```python
from rtmdk.pipeline import PipelineExecutor, EmbedStage, RetrieveStage

pipeline = PipelineExecutor([
    EmbedStage(embedder),
    RetrieveStage(field),
])
ctx = pipeline.run("What is the capital of France?", top_k=5)
print(ctx.to_dict()["stages"])
```

## Metrics Format

```json
{
  "query_text": "What is the capital of France?",
  "route": "fast",
  "top_k": 5,
  "results_count": 5,
  "explanations_count": 5,
  "stages": [
    {"stage": "embed", "latency_ms": 12.5, "input_count": 0, "output_count": 0},
    {"stage": "route", "latency_ms": 0.1, "input_count": 0, "output_count": 0},
    {"stage": "retrieve", "latency_ms": 0.8, "input_count": 0, "output_count": 5},
    {"stage": "rerank", "latency_ms": 1.2, "input_count": 5, "output_count": 5},
    {"stage": "calibrate", "latency_ms": 0.0, "input_count": 5, "output_count": 5},
    {"stage": "explain", "latency_ms": 0.3, "input_count": 5, "output_count": 5}
  ],
  "total_latency_ms": 14.9
}
```

## Graceful Degradation

Every stage implements `fallback(ctx, exc) → ctx`:

| Stage | Failure Mode | Fallback Behavior |
|-------|--------------|-------------------|
| Embed | embedder crash | Propagate error (unrecoverable) |
| Route | router exception | Default to `"standard"` route |
| Retrieve | field exception | Propagate error (unrecoverable) |
| Rerank | reranker exception | Skip reranking, keep original results |
| Calibrate | calibrator exception | Skip filtering, keep all results |
| Explain | explainer exception | Return results without explanations |

Failed stages are marked `degraded: true` in metrics but the pipeline continues.

## Health Checks

```python
health = memory.health_check_pipeline()
# {"healthy": True, "stages": [{"stage": "embed", "healthy": True, "reason": None}, ...]}
```

Each stage can override `health_check()` to implement component-specific probes.

## Prometheus Metrics

```python
from rtmdk.pipeline import to_prometheus_format

output = memory.retrieve_nodes_pipeline(query, embedding)
prom_text = to_prometheus_format(output["metrics"])
```

Example output:
```
# HELP rtmdk_query_latency_ms Total query latency
# TYPE rtmdk_query_latency_ms gauge
rtmdk_query_latency_ms{query="What is the capital of France?"} 14.9

# HELP rtmdk_stage_latency_ms Per-stage latency
# TYPE rtmdk_stage_latency_ms gauge
rtmdk_stage_latency_ms{stage="embed",error="0",degraded="0"} 12.5
rtmdk_stage_latency_ms{stage="route",error="0",degraded="0"} 0.1
```

## Backward Compatibility

- `retrieve_nodes()` — **preserved**, no breaking changes.
- `retrieve_nodes_with_explanations()` — **preserved**.
- `build_pipeline()` — new, opt-in.
- All existing configs work unchanged.

## Batch Execution

```python
from rtmdk.pipeline import BatchPipelineExecutor

batch = BatchPipelineExecutor(memory.build_pipeline().stages)
outputs = batch.run_batch(["q1", "q2", "q3"], top_k=5)
# outputs is a list of ctx.to_dict()
```

## Plugin Registry

Register and instantiate custom stages by name:

```python
from rtmdk.pipeline.registry import StageRegistry
from rtmdk.pipeline.base import PipelineStage

class MyRerankStage(PipelineStage):
    name = "my_rerank"
    def process(self, ctx):
        # ... custom logic ...
        return ctx

registry = StageRegistry()
registry.register("my_rerank", MyRerankStage)
stage = registry.create("my_rerank")
```

## Future Work

- Extract query cache, distributed lock, and query rewrite into separate stages.
- Entry-point discovery for third-party stage plugins.
- True vectorized batch retrieval in RetrieveStage.
