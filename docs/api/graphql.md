# GraphQL API

RTMDK ships an optional **Strawberry GraphQL** API mounted at `/graphql`
(when `strawberry-graphql` is installed — see the `graphql` extra in
`pyproject.toml`). The schema lives in `rtmdk/server/graphql_schema.py` and
mirrors the REST memory/pipeline operations plus subscriptions.

## Types

| Type | Fields |
|------|--------|
| `Health` | `status`, `version`, `memory_nodes` |
| `MemoryNode` | `id`, `content`, `salience`, `phase`, `amplitude` |
| `MemoryResult` | `node_id`, `score`, `content` |
| `PipelineResult` | `query`, `results`, `route`, `total`, `metrics` |
| `PipelineMetrics` | `stages` (per-stage latency / errors / degraded), `total_latency_ms`, `breaker_states` |

## Example Query

```graphql
query {
  health { status version memory_nodes }
}

query {
  pipelineQuery(query: "What is resonance?", topK: 5) {
    route
    total
    results { nodeId score content }
    metrics { totalLatencyMs stages { stage latencyMs degraded } }
  }
}
```

```bash
curl -X POST http://localhost:8080/graphql \
  -H "X-API-Key: rtmdk-local" \
  -H "Content-Type: application/json" \
  -d '{"query": "{ health { status memoryNodes } }"}'
```

The GraphiQL playground is available at `http://localhost:8080/graphql`
in the browser.

## Full Documentation

- Schema source: `rtmdk/server/graphql_schema.py`
- [API Reference — server & HTTP API section](../01_API_REFERENCE.md)
- Benchmark: `scripts/bench_graphql_websocket.py`
