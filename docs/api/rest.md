# REST API

The production server is a FastAPI app exposing **49 endpoints**. Start it with:

```bash
python -m rtmdk
# → http://localhost:8080 (RTMDK_PORT; project .env sets 8081)
```

## Authentication

All protected endpoints accept an API key via header:

```bash
curl -H "X-API-Key: rtmdk-local" http://localhost:8080/v1/memory/nodes
# or: Authorization: Bearer rtmdk-local
```

Default key is `rtmdk-local` — the server logs a security warning when auth is
enabled with the default key. Set `RTMDK_API_KEY` (e.g. in `.env`) for any
non-local deployment.

## Endpoint Groups

| Group | Examples |
|-------|----------|
| Health / Metrics | `GET /health`, `GET /health/deep`, `GET /metrics` (Prometheus) |
| OpenAI-compatible | `POST /v1/chat/completions`, `GET /v1/models`, `POST /v1/embeddings` |
| Memory query | `POST /v1/memory/query`, `/query_pipeline`, `/batch_query` |
| Pipeline ops | `GET /v1/memory/pipeline/health|metrics|plan|dag|stream` (SSE) |
| Node CRUD | `POST/GET/PUT/DELETE /v1/memory/nodes`, `POST /v1/memory/batch_ingest` |
| Import / Export | `GET /v1/memory/export`, `POST /v1/memory/import` |
| Analytics | `GET /v1/analytics/overview|memory|events|pipeline|report` |
| Admin | `/v1/admin/audit-log`, `/retention`, `/cache`, `/config`, `/api-keys`, `/tenants` |
| SOT | `GET /v1/sot/status`, `GET /v1/sot/vocab`, `POST /v1/sot/bootstrap` |
| Misc | `/v1/replication/*`, `/v1/webhooks`, `WS /ws/memory` |

## Example

```bash
curl -X POST http://localhost:8080/v1/memory/query \
  -H "X-API-Key: rtmdk-local" \
  -H "Content-Type: application/json" \
  -d '{"query": "What do I know about coffee?", "top_k": 5}'
```

## Full Documentation

- [API Reference — all 49 endpoints with request/response details](../01_API_REFERENCE.md)
- [Production Guide](../02_PRODUCTION_GUIDE.md)
