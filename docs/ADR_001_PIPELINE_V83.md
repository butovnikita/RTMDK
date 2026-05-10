# ADR 001: Pipeline v8.3 Architecture

> Status: Accepted
> Date: 2026-05-09
> Deciders: RTMDK Core Team

## Context

RTMDK v8.2 used a monolithic `retrieve_nodes()` method that combined embedding, routing, retrieval, reranking, calibration, and explanation into a single opaque function. This made it impossible to:

- Measure which step consumed the most time
- Skip expensive steps for simple queries
- Isolate failures to specific components
- A/B test changes to individual steps

## Decision

Replace the monolithic retrieval with an explicit 6-stage pipeline:

```
Embed → Route → Retrieve → Rerank → Calibrate → Explain
```

Each stage is an independent `PipelineStage` with:
- `process(ctx)` — main logic
- `fallback(ctx, exc)` — graceful degradation
- `circuit_breaker` — automatic fault isolation

## Consequences

### Positive

- **Observability**: Per-stage latency, error rate, breaker state
- **Optimization**: Query planner skips unnecessary stages (~40% latency reduction)
- **Reliability**: Circuit breakers prevent cascading failures
- **Extensibility**: New stages register via entry points
- **Testing**: Each stage tested independently

### Negative

- **Complexity**: 18 test files vs 1 previously
- **Overhead**: Pipeline context adds ~0.1ms per query
- **Migration**: Legacy `retrieve_nodes()` preserved but requires opt-in

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Keep monolithic | Simple | Unobservable, unoptimizable | Rejected |
| Microservices | Independent scaling | Network overhead, ops complexity | Rejected |
| **In-process stages** | Fast, observable, testable | Slightly more code | **Accepted** |

## Metrics

- Latency overhead: <0.1ms (measured)
- Test coverage: 147 pipeline tests (902 total)
- Production uptime: 100% (no breaker opens in normal operation)
