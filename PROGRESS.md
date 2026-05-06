# RTMDK Production Hardening — Progress Log

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

*Last updated: 2026-05-01*
