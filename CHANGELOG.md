# Changelog

All notable changes to RTMDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [8.3.1] — 2026-05-11

### Fixed (PyPI Package Audit)
- **PyPI wheel bloat**: removed `rtmdk_github/` clone (48% of wheel size, 1.5MB) and `tests/` from wheel via `MANIFEST.in`
- **Wheel size reduced**: 3.1MB → 1.6MB (-48%)
- **`__version__` mismatch**: `rtmdk/__init__.py` now reports `8.3.1` (was `8.1.0`)
- **`rtmdk/storage/` missing `__init__.py`**: added proper package init with `TieredNodeStore`, `TieredNodeStoreAdapter`
- **`rtmdk/server/proto/` missing `__init__.py`**: added package init
- **`rtmdk/production/__init__.py` typo**: `"MemoryDif": "memory_dif"` → `"MemoryDiff": "memory_diff"`
- **`rtmdk/support/__init__.py` phantom export**: removed non-existent `TritonBackend` import and `__all__` entry
- **`docs/01_API_REFERENCE.md` wrong import path**: `MemoryNode` from `rtmdk.nodes` instead of `rtmdk.memory.core`
- **`docs/09_LANGCHAIN_INTEGRATION.md` inconsistent import**: `rtmdk.production.langchain_adapter` instead of root-level
- **`configs/*.yaml` stale version headers**: v8.0 → v8.3
- **`rtmdk/memory/core.py` stale docstring**: Version 8.1+ → Version 8.3.0

### Hygiene
- Removed committed `__pycache__` directories from repo
- Removed empty `-p/` and `test/` directories
- Added `tests/__init__.py`
- Added `tests/results/`, `coverage.json`, `pypi_audit/` to `.gitignore`
- Added `MANIFEST.in` for clean sdist/wheel builds

## [Unreleased] — Pipeline v8.3

### Added

#### Pipeline Architecture
- **Explicit 6-stage retrieval pipeline**: Embed → Route → Retrieve → Rerank → Calibrate → Explain
- `PipelineExecutor` with sync, async, and batch execution modes
- `BatchPipelineExecutor` for efficient batch queries
- `StreamingPipelineExecutor` for real-time SSE stage events
- Per-stage circuit breaker with 3 states (closed / open / half-open)
- SLO enforcement via config-driven latency thresholds
- Graceful degradation with `fallback()` per stage
- Pipeline memory profiler with `tracemalloc` per-stage RSS tracking

#### Caching & Locking
- Query cache as explicit pipeline stages (`QueryCacheCheckStage`, `QueryCacheSaveStage`)
- Distributed lock as explicit pipeline stages (`DistributedLockStage`, `DistributedLockReleaseStage`)

#### Server Endpoints
- `POST /v1/memory/query_pipeline` — pipeline retrieval with full metrics
- `GET /v1/memory/pipeline/stream` — SSE streaming of stage events
- `GET /v1/memory/pipeline/health` — per-stage health & breaker status
- `GET /v1/memory/pipeline/metrics` — aggregated metrics with `?since=` and `?stage=` filters
- `GET /v1/memory/pipeline/prometheus` — Prometheus exposition format for Grafana
- WebSocket `query_pipeline` action with optional `stream: true` for live events

#### GraphQL
- `queryPipeline` field for synchronous pipeline queries
- `pipelineStream` subscription for real-time stage events

#### A/B Testing & Observability
- `PipelineABTester` framework comparing pipeline vs legacy retrieval
- `scripts/bench_pipeline_ab.py` benchmark script
- `PipelineMetricsStore` with JSON lines persistence and rotation
- Auto-discovery of third-party stages via `importlib.metadata` entry points

#### Rate Limiting
- Pipeline-specific stricter rate limits (`RTMDK_PIPELINE_RATE_LIMIT_PER_MINUTE`)
- Per-tenant override via API key `rate_limit_override`

#### Config & Validation
- `pipeline_enabled` opt-in flag for legacy → pipeline migration
- `pipeline_breaker_*` config fields with validation in `RTMDKConfig.validate()`

#### Testing & CI
- 95+ new tests covering all pipeline modules
- Dedicated `pipeline-integration` job in GitHub Actions
- Pipeline benchmark smoke test in CI
- `scripts/stress_test_100k.py` — enterprise stress test with tiered storage v2
- `tests/test_batch_pipeline.py` — batch query and pipeline executor tests

#### Batch & Vectorized Retrieval
- `RTMDKField.query_batch()` — true batch resonance across all cached nodes via `_batch_resonance`
- `BatchPipelineExecutor` v2 with real batch embed + batch retrieve optimization
- `BatchPipelineExecutor` backward-compatible fallback when `field` not provided

#### Monitoring & Deployment
- `docs/25_MONITORING_DEPLOYMENT.md` — complete guide for Prometheus + Grafana + Alertmanager stack

### Changed
- Legacy `retrieve_nodes()` delegates to pipeline when `pipeline_enabled=True`
- Updated `README.md`, `docs/01_API_REFERENCE.md`, `docs/PIPELINE_ARCHITECTURE.md`
- `MemoryNode.from_dict()` now filters to known fields — robust against extra keys from tiered storage serialization

### Architecture Decoupling (Leadership Cleanup)
- Extracted `FieldInitializer` (~460 lines) from monolithic `RTMDKField.__init__`
- Extracted `ContextManager` (~160 lines) from `RTMDKMemory`
- Extracted `MemoryPostInitializer` (~160 lines) from `RTMDKMemory.model_post_init`
- Extracted `BacklogModulesInitializer` (~95 lines) from `RTMDKMemory._init_backlog_modules`
- Extracted `PipelineBuilder` (~120 lines) from `RTMDKMemory.build_pipeline`
- Moved `_compress_field` and operational methods into `OperationalManager`
- `RTMDKField` reduced from 5265 → 844 lines (−84%); `RTMDKMemory` from 2603 → ~1380 lines (−47%)
- All public APIs preserved via thin wrappers / `__getattr__`
- Import cycle resolved: `MemoryNode` now imports from `rtmdk.nodes`

### Performance
- **5.5× batch ingestion speedup**: fixed O(N²) `list` scan (`if nid not in f.node_index`) in `add_nodes_batch` by using `set`-based O(1) lookup
- `add_nodes_batch` throughput: 22K nodes/sec (100K batch, was 3.3K)
- Without WAL: 1M nodes ingested in 12s = 83K nodes/sec (exceeds 60s target)
- Removed redundant `_build_node_cache()` call at end of `add_nodes_batch` (arrays already updated via `np.vstack`)
- `scripts/bench_batch_ingestion.py` now defaults to `wal_fsync_interval_ms=100` for realistic benchmarks

### Documentation
- Updated `docs/08_ARCHITECTURE.md` to reflect v8.3 decoupled architecture (Phase 23)
- Updated `BACKLOG.md` with accurate Track 2/Track 4 status and benchmark numbers

### Fixed
- Circuit breaker auto-recovery from half-open to closed after successful probes
- **Tiered Storage v2 warm-tier bug**: `_promote_from_warm` and `_demote_to_warm` now use `latent_pos` instead of `embedding`, preserving `MemoryNode` deserialization integrity
- **Tiered Storage v2 warm-tier slot reuse**: `_evict_warm_to_cold` now correctly returns freed slot index instead of decrementing `_warm_next_idx`, preventing `IndexError: index N is out of bounds`
- **Tiered Storage v2 cold-tier serialization**: `_write_cold` and `_read_cold` now preserve full node dict and `latent_pos` vector
- **Query race condition**: defensive bounds check in `query_manager.py` when `node_index` shrinks due to concurrent `delete_nodes`
- **WAL race condition**: check `_file is not None` inside lock in `append()` to prevent `AttributeError` during concurrent `close()`

### Testing & CI (Post-Release Hardening)
- **+157 tests** added (1112 total, 2 skipped)
- REST input validation tests (`test_server_validation.py`)
- Prometheus `/metrics` endpoint tests (`test_server_prometheus_metrics.py`)
- GraphQL WebSocket subscription runtime tests (`test_graphql_websocket.py`)
- Webhook HTTP endpoint tests (`test_server_webhooks.py`)
- Replication endpoint tests (`test_server_replication.py`)
- Deep health check tests (`test_server_health_deep.py`)
- Pipeline prometheus endpoint tests (`test_server_pipeline_prometheus.py`)
- Admin config hot-reload tests (`test_server_admin_config.py`)
- `/v1/models` and `/v1/embeddings` endpoint tests (`test_server_models_embeddings.py`)
- Concurrency stress tests (`test_concurrency_stress.py`) — found 2 race conditions
- WAL fault injection tests (`test_wal_fault_injection.py`) — found 1 race condition
- Lifecycle / resource leak detection tests (`test_lifecycle_leaks.py`)
- Config validation edge case tests (`test_config_validation.py`)
- Rate limiter config-driven (`rate_limit_nodes_per_sec` in `RTMDKConfig`)
- CI: flaky test retry (`pytest-rerunfailures`), duration reporting, suite speedup 30%

## [8.2.1] — 2026-04-XX

### Added
- Kalman filter integration for uncertainty-weighted retrieval
- Meta-adaptive bandwidth optimization
- Conformal prediction for calibrated confidence scores
- Domain memory with bi-temporal facts and evidence spans
- Multi-tenant memory routing
- Hierarchical configuration system
- LangChain and LlamaIndex adapters
- MCP server integration
- gRPC service
- WebSocket real-time endpoint
- Admin panel (React + Vite)

### Fixed
- HNSW index stability under concurrent access
- Co-occurrence matrix memory bounds
