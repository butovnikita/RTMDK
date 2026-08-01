# RTMDK — Agent Context

> This file exists to help coding agents (Claude, Kimi, etc.) understand the project quickly.

## Project
- **Name:** RTMDK (Resonance-Topological Memory)
- **Version:** 8.3.1
- **Language:** Python 3.10
- **OS:** Windows (primary dev), Linux (production Docker)
- **GPU:** CUDA 12.8, torch 2.12.0.dev

## Key Directories
```
rtmdk/memory/       — Core memory system (core.py, field.py, config.py)
rtmdk/memory/sot_v2/ — SOT v2.0 embedder (SIF + contrastive fine-tuning)
rtmdk/production/   — Production features (reranker, cascade router, BM25)
rtmdk/engines/      — Causal traversal, planners, evaluators
rtmdk/support/      — HNSW, projection, BM25, circuit breaker
rtmdk/server/       — FastAPI / GraphQL / WebSocket server
rtmdk/utils/        — JSON logging, async embedder
docs/               — Architecture, API reference, setup guides
tests/              — 1128 tests (pytest: 1126 passed, 2 skipped)
legacy/            — Frozen SillyTavern dev servers (rtmdk_server.py, st_proxy, ...)
scripts/            — Benchmarks and diagnostics
```

## Backlog Modules (v8.3.0)

### EngramEmbeddingCache (`rtmdk/memory/engram_cache.py`)
- **Purpose:** Hot/warm/cold tiered cache to avoid TieredNodeStore disk scans during engram retrieval.
- **Config:** `sot.engram_cache_enabled` (default `True`), `sot.engram_cache_max_hot` (10k), `sot.engram_cache_max_warm` (90k).
- **API:** `.add(nid, emb)`, `.get(nid)`, `.get_all()`, `.clear()`, `.save(path)`, `.load(path)`.
- **Wiring:** Auto-populated in `add_node()`, used in `_retrieve_and_format()` when `engram_manager` is active.

### Observability / Telemetry (`rtmdk/memory/observability.py`)
- **Purpose:** In-memory latency histograms (p50/p95/p99), cache hit/miss ratio, threshold alerting, Prometheus export.
- **Config:** `sot.observability_enabled` (default `False`).
- **Alert handlers:** `WebhookAlertHandler`, `SlackAlertHandler`, `PagerDutyAlertHandler`.
- **API:** `MemoryMetrics.record_query(latency_ms, cache_hit)`, `.to_prometheus()`, `.flush_to_file(path)`, `.check_alerts()`.
- **Wiring:** `retrieve_nodes()` records query latency; `add_node()` records ingestion latency.

### DistributedLock (`rtmdk/memory/distributed_lock.py`)
- **Purpose:** File-based or Redis inter-process lock with intra-process thread safety.
- **Config:** `sot.distributed_lock_path`, `sot.distributed_lock_backend` (`"file"` or `"redis"`), `sot.distributed_lock_redis_url`.
- **API:** `.acquire(blocking=True)`, `.release()`, context-manager compatible.
- **Wiring:** `retrieve_nodes()` acquires/releases around retrieval pipeline.

### RAG Quality (`rtmdk/memory/rag_quality.py`)
- **Purpose:** Query decomposition, sentence-level reranking, explicit feedback loop.
- **Config:**
  - `sot.sentence_reranker_enabled` (default `False`)
  - `sot.query_decomposition_enabled` (default `False`) — optional `llm_client` for advanced decomposition
  - `sot.feedback_loop_enabled` (default `False`)
  - `sot.feedback_loop_persist_path` — JSON persistence path
- **API:**
  - `QueryDecomposer.decompose(query)` → list of sub-queries
  - `SentenceReranker.rerank(query, results, top_k)` — batched sentence embedding
  - `FeedbackLoop.add_feedback(query, node_text, relevant)` → bool (requires SOTv2 embedder)
  - `FeedbackLoop.load()` / `_flush()` — persistence
- **Wiring:**
  - `QueryDecomposer` used in `_retrieve_and_format()`
  - `SentenceReranker` used in `retrieve_nodes()` after cache
  - `FeedbackLoop` accessible via `RTMDKMemory.add_feedback(query, node_id, relevant)`

### Explainability (`rtmdk/memory/explainability.py`)
- **ResultExplainer:** Human-readable explanations for why a memory was retrieved.
- **QueryRewriter:** Auto-rewrite query when top-score < threshold.
- **QueryIntentClassifier:** Classify intent as factual/exploratory/conversational/comparative.
- **Config:** `sot.result_explainability_enabled`, `sot.query_rewrite_enabled`, `sot.query_rewrite_threshold`, `sot.query_intent_classification_enabled`.
- **API:** `retrieve_nodes_with_explanations(query, embedding, ...)` → `{results, explanations, intent}`.

### Safety (`rtmdk/memory/safety.py`)
- **RollbackManager:** Capture snapshots and rollback memory to previous state.
- **PoisonedMemoryDetector:** Detect nodes with high out-degree, repetitive content, extreme sentiment.
- **API:** `memory.take_snapshot()`, `memory.rollback(timestamp)`, `memory.detect_poisoned_memories()`.

### Timeline & Narrative (`rtmdk/memory/timeline.py`)
- **MemoryTimeline:** Chronological view of memories per session/tier/time range.
- **MemoryNarrator:** Generate human-readable stories from episodic memories. Markdown export.
- **API:** `timeline.get_timeline(session_id=...)`, `narrator.narrate_session(session_id)`, `narrator.export_markdown(path)`.

### State Persistence
- `RTMDKMemory.save_state(dir_path)` — persists engram cache, feedback buffer, metrics snapshot.
- `RTMDKMemory.load_state(dir_path)` — restores engram cache, feedback loop.

### JSON Logging (`rtmdk/utils/json_logger.py`)
- Structured JSON logging for ELK/Loki integration.
- Usage: `setup_json_logging(level=logging.INFO)`.

### Async Embedder (`rtmdk/utils/async_embedder.py`)
- Async wrapper with request batching for high-throughput scenarios.
- Usage: `async_embedder = AsyncEmbedder(embed_fn, batch_size=16)`.

### Config Validation
- `RTMDKConfig.validate()` → list of warning strings for conflicting settings.
- Auto-called on initialization with `logger.warning`.

### Health Check
- `RTMDKMemory.health_check()` → `{"status": "healthy"|"degraded"|"unhealthy", "checks": {...}}`.
- Includes: node count, memory, disk, circuit breaker state, metrics snapshot.

### Circuit Breaker for Embedder
- `CircuitBreaker` from `rtmdk/support/circuit_breaker.py` wraps embedder.
- After 3 consecutive failures → returns zero vector and logs warning.
- Config: `embedder_circuit_breaker_enabled` (default `True`), `embedder_cb_threshold`, `embedder_cb_recovery`.

## Critical Constraints
1. **SIF OOM boundary:** Dense PMI matrix in `sif_embedder.fit()` scales as `O(n_valid²)`. For vocab > 5000, sparse PMI path (`scipy.sparse` + `TruncatedSVD`) activates automatically.
2. **Conformal invalidation:** After any embedder training (`train_sot_v2`), the conformal calibrator MUST be reset or coverage guarantees are void.
3. **Thread safety:** `add_node()` has `threading.Lock` for SOT online update buffer. `EngramEmbeddingCache` uses `RLock`.
4. **Config access:** Always use flat-field access (e.g., `config.engram_cache_enabled`) backed by `_FIELD_GROUPS` mapping.

## Testing
- Run full suite: `python -m pytest tests/ -x -q`
- New backlog tests: `python -m pytest tests/test_backlog_modules.py -v`
- Benchmark: `python scripts/bench_backlog_modules.py`
- Perf baseline: `benchmarks/baseline.json` (regenerate: `RTMDK_ADD_RATE_LIMIT=100000 python scripts/bench_pipeline_production.py --dataset datasets/qa_1000_en.json --output benchmarks/baseline.json`)

## Code Style (enforced in CI, blocking)
- `black rtmdk tests` (line-length 120, config in pyproject.toml; `legacy/` excluded as frozen)
- `flake8 rtmdk tests` (config `.flake8`: E203/W503 ignored for black-compat; core.py re-exports and generated proto excluded)
- Admin: `cd admin && npm run lint && npm run build`; SDK: `cd sdk/typescript && npm run lint && npm test`
- ALWAYS run black+flake8 before committing changes to `rtmdk/` or `tests/`

## Git Workflow
- Commits use conventional format: `feat(scope): description`, `fix(scope): description`, `refactor(scope): description`.
- Push to `main` after all tests pass.
