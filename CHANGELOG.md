# Changelog

All notable changes to RTMDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

### Fixed
- Circuit breaker auto-recovery from half-open to closed after successful probes
- **Tiered Storage v2 warm-tier bug**: `_promote_from_warm` and `_demote_to_warm` now use `latent_pos` instead of `embedding`, preserving `MemoryNode` deserialization integrity
- **Tiered Storage v2 warm-tier slot reuse**: `_evict_warm_to_cold` now correctly returns freed slot index instead of decrementing `_warm_next_idx`, preventing `IndexError: index N is out of bounds`
- **Tiered Storage v2 cold-tier serialization**: `_write_cold` and `_read_cold` now preserve full node dict and `latent_pos` vector

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
