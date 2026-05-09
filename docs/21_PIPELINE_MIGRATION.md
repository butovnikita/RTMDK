# Pipeline Migration Guide (v8.3+)

This guide helps you migrate from the legacy `retrieve_nodes()` API to the new explicit pipeline architecture.

## Overview

The pipeline architecture (v8.3+) provides:
- **Observability**: per-stage latency metrics, breaker states
- **Reliability**: circuit breakers, graceful degradation
- **Flexibility**: swappable stages, plugin registry
- **Streaming**: real-time SSE/WebSocket/GraphQL events

## Quick Start

### 1. Enable Pipeline (Opt-In)

```python
from rtmdk import RTMDKMemory, RTMDKConfig

config = RTMDKConfig(
    pipeline_enabled=True,  # ← delegates retrieve_nodes() to pipeline
)
mem = RTMDKMemory(config=config, embedder=embed_fn)
```

With `pipeline_enabled=True`, legacy `retrieve_nodes()` automatically uses the pipeline internally. No other code changes required.

### 2. Use Pipeline API Directly

```python
# New: full observability
result = mem.retrieve_nodes_pipeline("What is resonance?", top_k=5)

print(result["results"])      # ranked nodes
print(result["route"])        # "factual" | "standard" | "deep"
print(result["metrics"])      # per-stage latency
```

### 3. Check Pipeline Health

```bash
curl http://localhost:8080/v1/memory/pipeline/health
```

```json
{
  "overall": "healthy",
  "stages": [
    {"name": "embed", "enabled": true, "breaker_state": "closed"},
    {"name": "retrieve", "enabled": true, "breaker_state": "closed"}
  ]
}
```

## Stage Breakdown

| Stage | Purpose | Fallback Behavior |
|-------|---------|-------------------|
| Embed | Text → embedding | Returns None embedding |
| Route | Query classification | Uses "standard" route |
| Retrieve | Resonance/HNSW/BM25 | Returns empty results |
| Rerank | Sentence reranking | Skips reranking |
| Calibrate | Conformal filtering | Skips calibration |
| Explain | Result explanations | No explanations |

## Circuit Breaker Configuration

```python
config = RTMDKConfig(
    pipeline_breaker_enabled=True,
    pipeline_breaker_failure_threshold=5,
    pipeline_breaker_latency_violation_threshold=3,
    pipeline_breaker_recovery_timeout_ms=30000,
    pipeline_breaker_thresholds={
        "embed": 5000.0,     # ms
        "route": 100.0,
        "retrieve": 500.0,
        "rerank": 1000.0,
        "calibrate": 200.0,
        "explain": 100.0,
    },
)
```

## A/B Testing Before Full Migration

```python
from rtmdk.pipeline import PipelineABTester

tester = PipelineABTester(mem)
tester.compare_batch(["q1", "q2", "q3"], top_k=5)
print(tester.summary())
```

Or run the benchmark script:
```bash
python scripts/bench_pipeline_ab.py --queries 100 --nodes 500
```

## Rollback

If you encounter issues, simply set:
```python
pipeline_enabled=False  # or remove the flag
```

Legacy `retrieve_nodes()` will work exactly as before.

## Streaming for Real-Time UIs

### SSE
```bash
curl -N 'http://localhost:8080/v1/memory/pipeline/stream?query=hello&top_k=5'
```

### WebSocket
```javascript
ws.send(JSON.stringify({
    action: "query_pipeline",
    query: "hello",
    stream: true
}));
```

### GraphQL
```graphql
subscription {
  pipelineStream(query: "hello", topK: 5) {
    eventType
    stage
    latencyMs
  }
}
```

## Monitoring Checklist

- [ ] `GET /v1/memory/pipeline/health` — check breaker states
- [ ] `GET /v1/memory/pipeline/metrics` — latency trends
- [ ] `GET /v1/memory/pipeline/prometheus` — Grafana dashboard
- [ ] `GET /v1/analytics/pipeline` — dashboard overview
- [ ] CLI: `python -m rtmdk pipeline-diagnose`

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| All stages degraded | Breaker thresholds too low | Increase `pipeline_breaker_thresholds` |
| High latency | Profiler shows memory spikes | Reduce `top_k` or enable cache |
| Cache not hitting | Embedding mismatch | Ensure consistent embedder |
| Webhook spam | Stage flapping | Increase `recovery_timeout_ms` |
