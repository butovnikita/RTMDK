# RTMDK Production Hardening — Progress Log

## ✅ Pipeline v8.3 — Complete (50 commits, 871 tests)

**Status:** Production-ready. All features implemented, tested, documented.

### Features Delivered
| Feature | File | Tests |
|---------|------|-------|
| Explicit 6-stage pipeline | `rtmdk/pipeline/` | 8 integration |
| Graceful degradation | `base.py` | included |
| Health checks | `core.py` | included |
| Prometheus metrics | `metrics.py` | included |
| Batch execution | `batch.py` | 3 |
| Plugin registry | `registry.py` | 4 |
| Circuit breaker + SLO | `circuit_breaker.py`, `health.py` | 13 |
| Config-driven thresholds | `config.py` | 2 |
| Query cache stages | `cache_stages.py` | 5 |
| Distributed lock stages | `lock_stages.py` | 5 |
| Metrics persistence | `persistence.py` | 5 |
| Server endpoint | `server/app.py` | 4 |
| Pipeline migration (opt-in) | `core.py` | 4 |
| Entry-point discovery | `registry.py` | 3 |
| A/B testing framework | `ab_testing.py` | 5 |
| Benchmark script | `scripts/bench_pipeline_ab.py` | verified |
| Async execution | `executor.py`, `core.py` | 4 |
| Memory profiler | `profiler.py` | 4 |
| GraphQL pipeline query | `graphql_schema.py` | 2 |
| WebSocket pipeline query | `server/app.py` | 1 |
| WebSocket streaming stages | `server/app.py` | 1 |
| SSE streaming | `pipeline/streaming.py` | 5 |
| Pipeline health endpoint | `server/app.py` | 2 |
| Prometheus exposition | `server/app.py` | 2 |
| Metrics time/stage filtering | `persistence.py` | 2 |
| Pipeline config validation | `config.py` | 2 |
| GraphQL subscription | `graphql_schema.py` | 2 |
| Pipeline rate limiting | `tenant_rate_limiter.py` | 4 |
| Health monitor tests | `pipeline/health.py` | 8 |
| CI pipeline job | `.github/workflows/ci.yml` | verified |

### Statistics
- **871 passed, 1 skipped** — full regression suite
- **126 new tests** written in this branch
- **0 breaking changes**

### Documentation Updated
- `README.md` — Pipeline API section
- `docs/01_API_REFERENCE.md` — Section 14: Pipeline API
- `docs/PIPELINE_ARCHITECTURE.md` — Complete architecture guide

---

## ✅ 11. Pipeline v3: Circuit Breaker + SLO Enforcement (completed 2026-05-07)

**Goal:** Automatic fault isolation per stage. If a stage is too slow or failing, bypass it instead of crashing the pipeline.

### Changes Made
- **`rtmdk/pipeline/circuit_breaker.py`**: `CircuitBreaker` with 3 states (closed / open / half-open)
  - Opens after `failure_threshold` consecutive failures
  - Opens after `latency_violation_threshold` latency violations (> threshold)
  - Auto-recovery: transitions to half-open after `recovery_timeout_ms`, closes after successful probes
- **`rtmdk/pipeline/health.py`**: `PipelineHealthMonitor` manages per-stage SLO thresholds and breakers
- **`rtmdk/pipeline/base.py`**: `PipelineStage.run()` integrated with circuit breaker
  - Breaker open → skip `process()`, run `fallback()`, record `circuit_breaker_open` error
  - Breaker states tracked in `PipelineContext.breaker_states`
- **`rtmdk/memory/core.py`**: `build_pipeline()` attaches circuit breakers to all 6 stages with default thresholds
- **`tests/test_pipeline_circuit_breaker.py`**: 13 tests covering failure/latency open, half-open recovery, health monitor, integration

### Default SLO Thresholds
| Stage | Latency Threshold |
|-------|------------------|
| embed | 5000 ms |
| route | 100 ms |
| retrieve | 500 ms |
| rerank | 1000 ms |
| calibrate | 200 ms |
| explain | 100 ms |

### Test Results
- 13 new tests — all passing
- Full regression suite: **760 passed, 1 skipped**

---

## ✅ 14. Entry-Point Discovery + Docs Update (completed 2026-05-07)

**Goal:** Allow third-party packages to auto-register pipeline stages.

### Changes Made
- **`rtmdk/pipeline/registry.py`**: `StageRegistry.discover_entry_points()` 
  - Loads stages from setuptools entry points group `rtmdk.pipeline.stages`
  - Skips invalid classes and already-registered names
  - Python < 3.10 compatibility
- **`rtmdk/pipeline/__init__.py`**: Auto-discovery runs on import
- **`docs/PIPELINE_ARCHITECTURE.md`**: Added sections:
  - Pipeline Migration (opt-in retrieve_nodes → pipeline)
  - Query Cache & Distributed Lock as Stages
  - Entry-Point Discovery with pyproject.toml example
- **Tests**: 3 entry point discovery tests

### Test Results
- 3 new tests — all passing
- Full regression suite: **797 passed, 1 skipped**

---

## ✅ 13. Pipeline Metrics Dashboard + HTTP Endpoints (completed 2026-05-07)

**Goal:** Make pipeline metrics observable via HTTP and persistable for offline analysis.

### Changes Made
- **`rtmdk/pipeline/persistence.py`**: `PipelineMetricsStore` — append-only JSON lines with rotation
  - Thread-safe writes, summary statistics (mean/median/p95 per stage)
  - Integrated with `retrieve_nodes_pipeline(metrics_store=store)`
- **`rtmdk/server/app.py`**: 
  - `pipeline_metrics_store` global (optional)
  - `memory_query_pipeline` persists metrics when store configured
  - `GET /v1/memory/pipeline/metrics` — aggregated summary endpoint
- **`docs/PIPELINE_ARCHITECTURE.md`**: Added metrics persistence and HTTP endpoints sections
- **Tests**: 5 persistence tests + 2 server dashboard tests

### Test Results
- 7 new tests — all passing
- Full regression suite: **780 passed, 1 skipped**

---

## ✅ 12. Config-Driven Circuit Breaker Thresholds (completed 2026-05-07)

**Goal:** Remove hardcoded SLO thresholds from `build_pipeline()`.

### Changes Made
- **`rtmdk/memory/config.py`**: Added 6 new fields to `ProductionConfig`:
  - `pipeline_breaker_enabled` (default True)
  - `pipeline_breaker_failure_threshold` (default 5)
  - `pipeline_breaker_latency_violation_threshold` (default 3)
  - `pipeline_breaker_recovery_timeout_ms` (default 30_000)
  - `pipeline_breaker_half_open_max_calls` (default 3)
  - `pipeline_breaker_thresholds` (dict per stage, defaults provided)
- **`rtmdk/memory/core.py`**: `build_pipeline()` now reads all breaker settings from config
- **`tests/test_pipeline_circuit_breaker.py`**: 2 new tests verifying config read and disable

### Usage
```python
config = RTMDKConfig(
    pipeline_breaker_enabled=True,
    pipeline_breaker_failure_threshold=3,
    pipeline_breaker_thresholds={
        "embed": 2000.0,
        "retrieve": 100.0,
        "rerank": 500.0,
    },
)
```

### Test Results
- 2 new tests — all passing
- Full regression suite: **762 passed, 1 skipped**

---

## ✅ 10. Pipeline v2: Batch Execution + Plugin Registry (completed 2026-05-07)

**Goal:** Make the retrieval pipeline extensible and efficient for batch workloads.

### Changes Made
- **`rtmdk/pipeline/batch.py`**: `BatchPipelineExecutor` for sequential batch retrieval with shared stages
  - `BatchEmbedStage`: batch-aware embed stage that uses `embed_batch()` if available
- **`rtmdk/pipeline/registry.py`**: `StageRegistry` + `GLOBAL_REGISTRY`
  - Register custom stages by name: `registry.register("my_stage", MyStage)`
  - Instantiate by name: `registry.create("my_stage", **kwargs)`
  - Duplicate registration raises `ValueError`
- **`rtmdk/pipeline/__init__.py`**: Auto-registers all 6 default stages in `GLOBAL_REGISTRY`
- **`tests/test_pipeline_v2.py`**: 9 new tests covering batch execution, stage registry, and integration

### Usage
```python
from rtmdk.pipeline import BatchPipelineExecutor, GLOBAL_REGISTRY

# Batch retrieval
batch = BatchPipelineExecutor(memory.build_pipeline().stages)
outputs = batch.run_batch(["q1", "q2", "q3"], top_k=5)

# Plugin custom stage
from rtmdk.pipeline.registry import StageRegistry
class MyRerankStage(PipelineStage):
    name = "my_rerank"
    def process(self, ctx): ...

registry = StageRegistry()
registry.register("my_rerank", MyRerankStage)
```

### Test Results
- 9 new tests — all passing
- Full regression suite: **738 passed, 1 skipped**

---

## ✅ 9. Track 2: Tiered Storage (Hot / Warm / Cold) — Shipped

**Goal:** Support unlimited node count without proportional RAM growth.

### Implementation

- **`rtmdk/memory/tiered_storage.py`**: `TieredNodeStore` class
  - **Hot tier**: full `MemoryNode` objects in RAM (`OrderedDict` for LRU-friendly eviction)
  - **Warm tier**: dict serialization in RAM (~3–5× smaller footprint, no numpy overhead)
  - **Cold tier**: `msgpack+zlib` batches on disk, loaded lazily on access
  - LFU auto-promotion/demotion: least-frequently-used nodes demoted first
  - Thread-safe (`RLock`) for concurrent query / add / delete
  - Drop-in `MutableMapping` replacement for `self.nodes`

### Integration

- `RTMDKField.__init__`: instantiates `TieredNodeStore` when `tiered_storage_enabled=True`
- `_build_node_cache`: builds vectorized cache from hot + warm nodes only
- `query()`:
  1. Vectorized scan on hot+warm cache (`_query_vectorized`)
  2. Fallback to warm nodes via `_batch_resonance` if results < top_k
  3. Sampled cold fallback (random sample, batch load, resonance) if still insufficient
- `export_field` / `import_field`: serialize/deserialize `tiered_store` metadata + all node dicts

### Config

```python
config = RTMDKConfig(
    tiered_storage_enabled=True,
    tiered_storage_path="./rtmdk_cold_storage",
    tiered_hot_pct=0.01,
    tiered_warm_pct=0.09,
    max_nodes=100_000,
)
```

### Test Results

- 8 new tests in `tests/test_tiered_storage.py` — all passing
- Full regression suite: **340 passed, 2 skipped** (no regressions)

### Notes

- 1M-node benchmark deferred to v8.3 stress-test suite
- Warm tier uses dict form (not `numpy.memmap` as originally spec'd) — simpler, portable, still ~3–5× RAM savings
- Cold tier batch size is dynamic (up to `warm_limit` nodes per batch)

---

## Sprint: Accuracy Gap + Async Consolidation + Stability + Docs Cleanup

---

## ✅ 1. Accuracy Gap Closed (R@1: 87.1% → 94.0%)

**Goal:** Close the 10% accuracy gap between RTMDK and FAISS/BM25 on 1K-node benchmark.

### Results
| Metric | Before | After | Δ |
|--------|--------|-------|---|
| R@1 | 87.1% | **94.0%** | +6.9% |
| R@5 | 92.3% | **96.0%** | +3.7% |
| Exact Match | 94.5% | **96.7%** | +2.2% |
| P95 Latency | 10 ms | **0 ms** | −10 ms |
| Gap vs FAISS | 10.0% | **3.1%** | −6.9% |

### Root Causes Identified
1. **HNSW hard truncation**: Only `top_k * 3 = 45` candidates evaluated out of 1000 → true matches excluded.
2. **Random projection 768d→64d without normalization**: Destroyed cosine geometry.
3. **L2 HNSW on cosine embeddings**: Wrong metric for nomic-embed-text-v1.5.
4. **Phase coupling noise**: Random phases added 0.7×–1.0× multiplicative noise.
5. **Resonance kernel bug**: `"gaussian_phase"` fell through to `exp(-dist/bw)` instead of `exp(-dist²/(2*bw²))`.

### Changes Made
- `rtmdk/memory/core.py`: `_project()` now normalizes embeddings before random projection.
- `rtmdk/memory/core.py`: `_resonance_response()` now correctly handles `"gaussian_phase"` as Gaussian kernel.
- `rtmdk/memory/core.py`: HNSW pre-filtering threshold raised from `>500` to `>5000` nodes; candidate multiplier increased from `top_k*3` to `max(top_k*10, 200)`.
- `tests/test_rtmdk_v8_benchmark.py`: Uses optimized config (`latent_dim=128`, `resonance_kernel="cosine"`, `phase_coupling=0.0`, `min_response=0.001`).

### Notes
- For small datasets (<5K nodes), full vectorized scan is **faster and more accurate** than HNSW.
- `learn_projection=True` (IncPCA) **degrades** retrieval because old nodes keep stale projections while new nodes use updated matrix → mixed geometry space. Disabled for now.
- `production` preset with conformal/adaptive_bandwidth/kalman gives **57.5% R@1** — these modules need calibration before they help. Investigate separately.

---

## ✅ 2. Production Preset Debug

**Status:** `production` preset yielded 57.5% R@1 vs 94.0% for optimized config (on 1000 records).

### Root Cause
**`adaptive_bandwidth=True`** — per-node bandwidth computed from k-NN distances becomes unstable on small-to-medium fields (<10K nodes). Extreme bw factors (very small in dense regions, very large in sparse regions) completely distort resonance scores and destroy ranking accuracy.

### A/B Test Results
| Config | R@1 (1000 records) |
|--------|-------------------|
| Baseline (optimized) | **94.0%** |
| Production preset (before fix) | **57.5%** |
| Production + adaptive_bandwidth=False | **94.2%** |
| Production (after fix) | **95.6%** |

### Fix
- Disabled `adaptive_bandwidth` by default in `_production()` preset.
- Added comment explaining why: "unstable on small-to-medium fields (<10K nodes)".

### Notes
- `adaptive_bandwidth` may be re-enabled after stabilization (clipping, median smoothing, min-sample threshold).
- All other modules (`conformal`, `kalman`, `engrams`, `causal`, `ssm`, `trust`) have negligible impact on retrieval accuracy when properly configured.

---

## ✅ 3. Async Consolidation

**Goal:** `consolidate()` must not block the main thread.

**Current State:**
- `consolidate()` runs synchronously via `circuit_breaker.call()` in `step()`.
- Blocks for 75–95 seconds on ~1000 nodes.
- Mitigation: frequency reduced from every 20 steps to every 50 steps for <10K nodes.

**Implementation:**
- Added `consolidation_async: bool = False` to `CoreConfig` (env: `RTMDK_CONSOLIDATION_ASYNC`).
- Added `ThreadPoolExecutor(max_workers=1)` to `RTMDKField`.
- `step()` now submits consolidation to executor when `consolidation_async=True`.
- `step()` returns immediately — no blocking.
- `consolidate()` still uses `@_locked` in background thread; `add_node()` waits briefly if consolidation is active.
- Thread-safety fixes: `_build_node_cache()` compacts `node_index`, `query()` fallback loop skips deleted nodes.
- Added `shutdown()` for graceful executor shutdown.

**Benchmark Result (async enabled):**
- R@1: **95.6%** (vs 95.5% sync — no degradation)
- Exact: **96.8%** (matches FAISS)
- `step()` latency: **0 ms** (was 75–95 s)

---

---

## ✅ 4. Stability at 10K+ Nodes (completed 2026-05-06)

**Goal:** Verify latency, RAM, serialization at scale.

### Results

| Metric | 1K (baseline) | 5K | 10K |
|--------|--------------|-----|-----|
| **Insert throughput** | ~10K nodes/sec | 7,877 nodes/sec | 7,085 nodes/sec |
| **RAM** | ~16 MB | 299 MB (60 MB/1K) | 333 MB (33 MB/1K) |
| **Query P50** | <1 ms | 0.96 ms | **1.21 ms** |
| **Query P95** | <1 ms | 1.36 ms | **1.65 ms** |
| **Query P99** | <1 ms | 8.14 ms | **1.89 ms** |
| **Consolidation** | 0.3s | 0.3s | 0.8s |
| **Save** | — | 0.7s | 1.4s |
| **Load** | — | 0.3s | 0.5s |
| **Disk size** | — | 3.5 MB | 6.6 MB |
| **R@1** | 95.6% | 100%* | 100%* |

\* R@1 measured on synthetic variants (high semantic similarity); real-world accuracy expected ~95%.

### Critical Fix: HNSW Query Path

**Problem:** At N=10K, query latency exploded to **128ms P50**.
- Root cause: HNSW path fell through to a Python `for` loop calling `_resonance_response()` per candidate.
- 200 candidates × ~0.6ms Python-loop resonance = ~120ms per query.

**Fix (core.py):** HNSW path now uses `_batch_resonance()` — vectorized numpy computation on candidates:
```python
scores = self._batch_resonance(
    query_latent[np.newaxis, :],
    np.array([phase], dtype=np.float32),
    candidate_ids,
)[0]
```
- Result: 128ms → **1.21ms** (100× speedup).

**Secondary fix:** `_batch_resonance_numpy` had a scalar-bw broadcasting bug (`bw[np.newaxis, :]` on float). Fixed with `np.ndim(bw) == 0` check.

---

## ✅ 5. Docs Cleanup (completed 2026-05-01)

**Goal:** Remove overpromising claims (Raft, distributed sharding, PQ, 98% R@1, Byzantine) from docs.

**Files updated:**
- `docs/06_SCIENTIFIC_ARTICLE.md` — 98% → 95.6% throughout, PQ-64/Byzantine marked as planned
- `docs/07_DIALOGUE_EXPORT.md` — updated scaling table to v8.1 numbers, added disclaimers
- `docs/05_FINE_TUNING.md` — enterprise preset marked as roadmap
- `docs/08_ARCHITECTURE.md` — TrustConsensus explicitly marked as research prototype
- `docs/ROADMAP.md` — 98% → 95.6%, removed "outperforms by 16pp" claims
- `docs/FULL_AUDIT.md` — 98% → 95.6%

---

## ✅ 6. Sprint 4: CI/CD + PyPI Packaging (completed 2026-05-01)

**Goal:** Make RTMDK installable from PyPI and add automated testing.

### Changes Made
- **`pyproject.toml`**: Added `[build-system]` (setuptools), package metadata (name=rtmdk, version=8.2.0), pytest markers (`slow`), black/isort config.
- **`.github/workflows/ci.yml`**: Lint, type-check, test with multi-Python matrix (3.10/3.11/3.12).
- **`.github/workflows/publish.yml`**: Automated PyPI publish on git tags.
- **`rtmdk/memory/config.py`**: Added `adaptive_bandwidth_min_n` (default 50) — configurable minimum node count for adaptive bandwidth cache build.
- **`rtmdk/memory/field.py`**: Changed adaptive bw condition from `n > max(k, 50)` to `n >= max(k, adaptive_bandwidth_min_n)`.
- **`tests/test_local_bandwidth.py`**: Updated to use `adaptive_bandwidth_min_n=5` for small-N unit tests.

### Build Verification
```bash
python -m build
# Produced: rtmdk-8.2.0.tar.gz + rtmdk-8.2.0-py3-none-any.whl
```

### Test Results
- 265 passed, 1 skipped, 6 warnings (1 flaky rate-limit test)
- `pytest -m "not slow"` runs in ~4 seconds

### Known Issues (Pre-Deep Dive)
- **Git remote not configured** — commits are local-only. Push requires `git remote add origin <URL>`.
- **SOT benchmark**: R@1 = 9% on 1K QA — remains experimental, disabled by default.
- **adaptive_bandwidth**: Stabilized but ~56% R@1 vs 95.6% baseline — remains disabled by default.

---

## ✅ 7. Deep Dive: Dead Modules Investigation (completed 2026-05-01)

### 7.1 adaptive_bandwidth — "Stabilized to Death"

**Finding:** Current stabilization (clip 0.2–5.0 + [0.1×, 10×] global bounds) produces BW spread of only **1.4×**. The feature is effectively global bandwidth in disguise.

**Root cause (discovered in full research):** Curse of dimensionality. In 384d normalized embedding space, k-NN distances have CV < 0.5% and p99/p01 ratio ≈ 1.02×. Any transform of nearly-constant values produces nearly-constant bandwidth factors. Grid search of 252 configs confirmed: **no transform/clip/k combination produces meaningful adaptation without destroying accuracy**.

**Tests performed:**
| Benchmark | Global BW R@1 | Adaptive BW R@1 |
|-----------|---------------|-----------------|
| Synthetic clustered (128d) | 100% | 100% |
| SBERT semantic (384d, 300 QA) | 74.0% | 74.0% |

**Conclusion:** P1.2 local adaptive bandwidth (k-NN distance based) is mathematically doomed in high-D normalized spaces. The 56% R@1 figure from Sprint 3.2 was measured **before** stabilization.

**Decision:** Remove P1.2 local adaptive bandwidth entirely. Replace with original v8 **MetaAdaptiveKernel** (global kurtosis-based adaptive bandwidth) which was already implemented but disabled.

### 7.2 SOT (Self-Organizing Tokenizer) — "Benchmark was Broken"

**Finding:** Previous `test_sot_benchmark.py` was incorrect. It added nodes with **random dummy embeddings** but queried with **SOT embeddings** — different embedding spaces produced ~9% R@1 (effectively random).

**Corrected benchmark (both paths use SOT):**
| Method | R@1 (200 QA) |
|--------|--------------|
| SOT (byte tokenizer + SBERT bootstrap) | **73%** |
| SBERT baseline (external embedder) | **100%** |

**Conclusion:** SOT works as a lightweight fallback. 73% R@1 is acceptable for a zero-dependency embedder, though it cannot match full SBERT.

**Decision:** Keep SOT as experimental fallback. Corrected `tests/test_sot_benchmark.py` now validates >=55% R@1 against SBERT baseline.

---

## ✅ 8. Fix adaptive_bandwidth: Remove P1.2, Enable MetaAdaptiveKernel (completed 2026-05-01)

**Problem:** Two different features shared the name "adaptive bandwidth":
1. **P1.2 Local Adaptive BW** (`config.adaptive_bandwidth`) — k-NN distance based, broken in high-D
2. **MetaAdaptiveKernel** (`config.meta_adaptive`) — global kurtosis-based, working but disabled

**Changes Made:**
- **Removed P1.2 completely:**
  - Deleted `adaptive_bandwidth`, `adaptive_bandwidth_k`, `adaptive_bandwidth_min_n` from `CoreConfig`
  - Removed `_cached_bw`, k-NN distance computation, and all P1.2 code from `field.py` (~50 lines removed)
  - Deleted `tests/test_local_bandwidth.py`
  - Removed `adaptive_bandwidth` references from `test_config_matrix.py` and `test_sot_benchmark.py`
- **Enabled MetaAdaptiveKernel in production preset:** `meta_adaptive=True` in `_production()`
- **Created `tests/test_meta_adaptive.py`** — validates kurtosis-driven bandwidth adaptation

**MetaAdaptiveKernel verified:**
| Config | Final BW | R@1 | Status |
|--------|----------|-----|--------|
| Global bw=1.0 | 1.000 | 74.0% | baseline |
| MetaAdaptive (default) | 1.051 | 74.0% | ✅ safe |
| MetaAdaptive (aggressive) | 1.219 | 74.0% | ✅ safe |

**Test results:** 264 passed, 1 skipped, 6 warnings (1 flaky rate-limit test)

---

*Last updated: 2026-05-07*
