# RTMDK — Agent Context

> This file exists to help coding agents (Claude, Kimi, etc.) understand the project quickly.

## Project
- **Name:** RTMDK (Resonance-Topological Memory)
- **Version:** 8.2.1
- **Language:** Python 3.10
- **OS:** Windows (primary dev), Linux (production Docker)
- **GPU:** CUDA 12.8, torch 2.12.0.dev

## Key Directories
```
rtmdk/memory/       — Core memory system (core.py, field.py, config.py)
rtmdk/memory/sot_v2/ — SOT v2.0 embedder (SIF + contrastive fine-tuning)
rtmdk/production/   — Production features (reranker, cascade router, BM25)
rtmdk/engines/      — Causal traversal, planners, evaluators
rtmdk/support/      — HNSW, projection, BM25
rtmdk/server/       — FastAPI / GraphQL / WebSocket server
docs/               — Architecture, API reference, setup guides
tests/              — 738 tests (pytest)
scripts/            — Benchmarks and diagnostics
```

## Backlog Modules (v8.2.1)

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

### State Persistence
- `RTMDKMemory.save_state(dir_path)` — persists engram cache, feedback buffer, metrics snapshot.
- `RTMDKMemory.load_state(dir_path)` — restores engram cache, feedback loop.

## Critical Constraints
1. **SIF OOM boundary:** Dense PMI matrix in `sif_embedder.fit()` scales as `O(n_valid²)`. For vocab > 5000, sparse PMI path (`scipy.sparse` + `TruncatedSVD`) activates automatically.
2. **Conformal invalidation:** After any embedder training (`train_sot_v2`), the conformal calibrator MUST be reset or coverage guarantees are void.
3. **Thread safety:** `add_node()` has `threading.Lock` for SOT online update buffer. `EngramEmbeddingCache` uses `RLock`.
4. **Config access:** Always use flat-field access (e.g., `config.engram_cache_enabled`) backed by `_FIELD_GROUPS` mapping.

## Testing
- Run full suite: `python -m pytest tests/ -x -q`
- New backlog tests: `python -m pytest tests/test_backlog_modules.py -v`
- Benchmark: `python scripts/bench_backlog_modules.py`

## Git Workflow
- Commits use conventional format: `feat(scope): description`, `fix(scope): description`, `refactor(scope): description`.
- Push to `main` after all tests pass.
