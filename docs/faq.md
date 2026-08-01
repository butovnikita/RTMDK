# FAQ

## How do I start the server, and which port does it use?

```bash
python -m rtmdk
```

Uvicorn starts on `RTMDK_PORT` (default **8080**; the project `.env` sets
**8081**). Verify with `curl http://localhost:8080/health`.
See [Local Setup](03_LOCAL_SETUP.md).

## How does authentication work?

Protected endpoints require an API key via `X-API-Key: <key>` or
`Authorization: Bearer <key>`. The default key is `rtmdk-local`; the server
logs a security warning if auth is enabled with the default key. Set
`RTMDK_API_KEY` for anything beyond local use. Keys can be managed via
`/v1/admin/api-keys`. See [REST API](api/rest.md).

## Where is my memory stored on disk?

By default at `~/.rtmdk/memory.json` — despite the extension it is
**msgpack + zlib** compressed, not plain JSON. SOT checkpoints live alongside
(`~/.rtmdk/sot_checkpoint.json`). Use `/v1/memory/export` / `/import` for
portable snapshots.

## Do I need LM Studio (or any external LLM)?

No. LM Studio is optional — needed only if you want the OpenAI-compatible
chat endpoint (`/v1/chat/completions`) backed by a real local LLM. Memory,
embeddings (SOT v2.0 is self-contained), and retrieval work without it.

## How is RTMDK different from plain RAG?

RAG does cosine/dot-product ANN lookup over static vectors. RTMDK models
memory as a **dynamic cognitive field**: nodes are oscillators with phase,
amplitude, salience, and causal links; retrieval is resonance interference.
It also adds decay/forgetting, consolidation, pipeline observability, and
circuit breakers. Full comparison: [RTMDK vs SOTA RAG](24_RAG_COMPARISON.md).

## Does RTMDK load `.env` automatically?

Yes — since 2026-08 the entry points `python -m rtmdk` and
`start_production.py` auto-load `.env` via `python-dotenv` (real env vars take
priority). A bare library import of `rtmdk.server.app` does **not** load
`.env`, protecting test environments.

## How do I configure RTMDK without editing code?

Pick one of 9 presets (`local`, `production`, `research`, ...) and override
any of the 59 parameters via `RTMDK_*` env vars, e.g.
`RTMDK_PRESET=production RTMDK_TOP_K=10 python -m rtmdk`.
See [Configuration](getting-started/configuration.md) and the
[Fine-Tuning Guide](05_FINE_TUNING.md).

## Is the legacy `retrieve_nodes()` API still supported?

Yes. The v8.3 pipeline is opt-in (`pipeline_enabled=True`); when enabled, the
legacy method delegates to the pipeline internally with no other code changes.
See the [Pipeline Migration Guide](21_PIPELINE_MIGRATION.md).
