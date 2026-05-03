# RTMDK Mathematical Enhancements Backlog

> Prioritized by production impact × implementation feasibility.
> Estimates assume 1 senior engineer, familiar with the codebase.

---

## P0 — Critical (fix fundamental flaws)

### P0.1 Riemannian SGD on Poincaré Ball
**Owner:** TBD | **Estimate:** 3 days | **Status:** ✅ Completed (2026-05-01)

**Problem:** `consolidate()` applies Euclidean gradients to nodes living on a negatively-curved manifold. Nodes hit the ball boundary and are clamped, destroying the hyperbolic metric structure.

**Mathematics:**
- Riemannian gradient: $\nabla_R f(x) = (1/\lambda_x^2) \nabla_E f(x)$, where $\lambda_x = \frac{2}{1 - \|x\|^2/R^2}$
- Update: $x_{new} = \exp_x(-\eta \cdot \nabla_R f(x))$ via Möbius addition
- Midpoint: $m = \exp_a(\tfrac{1}{2} \log_a(b))$

**Implementation:**
- Fixed `exp_map_poincare` in `rtmdk/memory/geometry.py` and `rtmdk/utils/hyperbolic.py` — scalar factor was missing `ball_radius` multiplier, causing round-trip errors.
- Added `poincare_midpoint()` and `mobius_scalar_mul()` to `geometry.py`.
- Refactored `consolidate()` in `core.py` to use `poincare_midpoint()` for spatial merge when `hyperbolic=True`.
- Refactored `step()` attraction update to use Riemannian SGD (`conformal * grad_e` + `exp_map_poincare`).
- Added hyperbolic clamping in `add_node()` for `projection_learner` path (was missing).
- Removed stale local copies of `poincare_dist`, `exp_map_poincare`, `log_map_poincare`, `mobius_add` from `core.py`.
- Fixed `poincare_dist` in `geometry.py` / `hyperbolic.py` to multiply by `ball_radius` (was missing for R≠1).

**Expected Impact:**
| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Boundary-clamped nodes | 5–15% after consolidate | <1% | Count `\|x\| > 0.99R` |
| Silhouette score (k=5) | ~0.42 | >0.60 | `sklearn.metrics.silhouette_score` on latent positions |
| Position drift (std) | ~0.12 | <0.05 | Std of $\|x_{t} - x_{t-1}\|$ across consolidations |

**Risks:**
- Riemannian gradients are ~2× slower to compute (conformal factor + Möbius ops).
- May change "feel" of retrieval if users rely on current (buggy) clustering geometry.

**Acceptance Criteria:**
- [x] `exp_map_poincare` used in all position updates inside `consolidate()`
- [x] Unit test: node initialized at origin, gradient pushes it, position stays inside ball without clamping
- [x] Benchmark: consolidate on 1K nodes does not regress wall-time by >20%

---

## P1 — High Value (production quality)

### P1.1 Conformal Prediction for Retrieval Confidence
**Owner:** TBD | **Estimate:** 3 days | **Status:** ✅ Completed (2026-05-01)

**Problem:** `query()` returns scores, but there is no statistical guarantee that the top result is relevant. In production this means LLM may receive garbage context with high "confidence."

**Mathematics:**
- Calibration set: $\{(x_i, y_i, s_i)\}_{i=1}^{n}$ where $s_i = \text{resonance}(x_i, y_i)$
- Non-conformity score: $\alpha_i = 1 - s_i$ (lower = more conforming)
- Quantile: $\hat{q} = \lceil (n+1)(1-\alpha) \rceil / n$ quantile of $\{\alpha_i\}$
- Prediction set: $C(x_{n+1}) = \{y : s(x_{n+1}, y) \geq 1 - \hat{q}\}$

**Expected Impact:**
| Metric | Current | Target |
|--------|---------|--------|
| Retrieval coverage | N/A (no guarantee) | $\geq 90\%$ at $\alpha=0.10$ |
| Precision@K (with rejection) | baseline | +15–25% relative |
| False-context injections to LLM | unknown | measurable reduction |

**Risks:**
- Requires calibration set (~200 labeled query-result pairs). Cold-start issue.
- If distribution shifts, coverage drops until recalibration.

**Acceptance Criteria:**
- [ ] `query()` returns `confidence: float` and `prediction_set: List[str]` alongside results
- [ ] `RTMDKConfig` flag: `conformal_prediction: bool = False` (opt-in)
- [ ] Unit test: on synthetic data, empirical coverage $\geq 1 - \alpha - 0.05$

---

### P1.2 Local Adaptive Bandwidth (k-NN KDE)
**Owner:** TBD | **Estimate:** 1 day | **Status:** ✅ Completed (2026-05-01)

**Problem:** Global `bandwidth` blurs dense clusters and misses sparse concepts. Meta-adaptive kernel adjusts globally, not locally.

**Mathematics:**
$$\sigma_i = \sigma_{\text{global}} \cdot \left( \frac{\text{kdist}_i}{\text{median}(\text{kdist})} \right)^{1/2}$$
where $\text{kdist}_i$ = distance to $k$-th nearest neighbor ($k=5$).

**Expected Impact:**
| Metric | Current | Target |
|--------|---------|--------|
| Precision@10 (dense region) | ~0.62 | >0.75 |
| Recall@10 (sparse region) | ~0.45 | >0.60 |
| Mean reciprocal rank (MRR) | baseline | +10–15% relative |

**Risks:**
- k-NN computation is $O(N \log N)$ per cache rebuild; acceptable if cached.
- Very sensitive to $k$ choice; default $k=5$ should work for 64D.

**Implementation:**
- `_build_node_cache()` precomputes k-NN distances via `cKDTree` when `adaptive_bandwidth=True`.
- `_compute_resonance_chunk()` handles scalar and per-node `bw` arrays via `np.isscalar`/`np.maximum`.
- `_cached_bw` correctly sliced during session filtering.
- 10 unit tests covering median normalization, dense-vs-sparse bandwidth, cache invalidation, and query correctness.

**Acceptance Criteria:**
- [x] `_build_node_cache()` precomputes `kdist` array
- [x] `_compute_resonance_chunk()` uses per-node `bw` vector
- [x] No regression on `test_chunked_query_matches_non_chunked`

---

## P2 — Medium Value (observability & stability)

### P2.1 Spectral Graph Laplacian for Consolidation
**Owner:** TBD | **Estimate:** 2 days | **Status:** ✅ Completed (2026-05-01)

**Problem:** Greedy pairwise merge in `consolidate()` is $O(N^2)$ and falls into local optima. Does not discover global cluster structure.

**Mathematics:**
- Affinity: $W_{ij} = \exp(-d_{ij}^2 / 2\sigma^2) \cdot \frac{1 + \cos(\phi_i - \phi_j)}{2}$
- $L_{sym} = I - D^{-1/2} W D^{-1/2}$
- Bottom-$k$ eigenvectors → spectral embedding → k-means
- Eigengap heuristic: $k^* = \arg\max_k (\lambda_{k+1} - \lambda_k)$

**Expected Impact:**
| Metric | Current | Target |
|--------|---------|--------|
| Consolidate wall-time (1K nodes) | ~120ms | <200ms (sparse eigensolver) |
| Cluster purity (tier-based) | ~0.58 | >0.70 |
| Merge quality (manual sample) | ad-hoc | >80% "correct" merges |

**Risks:**
- `scipy.sparse.linalg.eigsh` may fail on poorly conditioned $L$.
- Changes merge behavior significantly; needs careful validation.

**Implementation:**
- `spectral_cluster_nodes()` in `rtmdk/memory/spectral.py` builds affinity, normalized Laplacian, bottom-k eigenvectors, and k-means++ clustering.
- Eigengap heuristic selects `k*` capped at `spectral_max_clusters`.
- `_spectral_merge_clusters()` in `RTMDKField` performs greedy pairwise merge within each discovered cluster.
- 13 unit tests: affinity symmetry, Laplacian eigenvalue bounds, eigengap, k-means, timeout fallback, integration.

**Acceptance Criteria:**
- [x] Fallback to greedy merge if eigendecomposition fails or timeout >500ms
- [x] `test_domain_memory` backward compatibility preserved
- [x] Eigengap auto-$k$ capped at `max_consolidation_clusters` config

---

### P2.2 Kalman Filtering on Manifold (Riemannian EKF)
**Owner:** TBD | **Estimate:** 4 days | **Status:** ✅ Completed (2026-05-01)

**Problem:** `ode_dynamics` and `consolidate()` apply ad-hoc noise. No principled uncertainty tracking for node positions.

**Mathematics:**
- Prediction: $\hat{x}_{t|t-1} = \exp_{x_{t-1}}(v_{t-1} \Delta t)$
- Innovation: $y_t = \log_{\hat{x}}(z_t)$
- Kalman gain on tangent space: $K_t = \Sigma H^T (H \Sigma H^T + R)^{-1}$
- Update: $x_{t|t} = \exp_{\hat{x}}(K_t y_t)$

**Expected Impact:**
| Metric | Current | Target |
|--------|---------|--------|
| Position trajectory smoothness | noisy | visually smooth |
| Outlier robustness (injected bad embedding) | corrupts node | absorbed, uncertainty grows |
| Uncertainty estimate | N/A | available per node |

**Risks:**
- High complexity; easy to introduce numerical instabilities.
- Covariance matrix per node increases memory by $O(d^2)$ per node (64×64 = 16KB per node × 10K nodes = 160MB).

**Implementation:**
- `KalmanFilter` in `rtmdk/memory/kalman.py` with diagonal and full-matrix modes, hyperbolic support via `log_map`/`exp_map`.
- `MemoryNode.covariance` serialized in `to_dict`/`from_dict`.
- Initialized on `add_node`; prediction + update + merge in `_do_merge()`; retrieval weighting in `query()`.
- 15 unit tests: init shapes, prediction growth, update shrinkage, position shift, merge information gain, query weighting, hyperbolic stay-in-ball.

**Acceptance Criteria:**
- [x] Optional feature: `enable_kalman_filter: bool = False`
- [x] Uncertainty used to weight retrieval: `score *= 1 / (1 + \text{tr}(\Sigma))`
- [x] Memory overhead capped: store diagonal only if `kalman_diagonal_approx: bool = True`

---

## P3 — Low Value / High Complexity (research tier)

### P3.1 Wasserstein-2 Distance for Field Stability
**Owner:** TBD | **Estimate:** 2 days | **Status:** 🔴 Not started

**Problem:** No quantitative metric for "how much did the field change?"

**Mathematics:**
- Discrete measure: $\mu = \sum_i w_i \delta_{x_i}$, $w_i \propto \text{amplitude}_i \cdot \text{salience}_i$
- $W_2(\mu, \nu)$ via linear assignment (`scipy.optimize.linear_sum_assignment`) or sinkhorn iteration

**Expected Impact:**
| Metric | Current | Target |
|--------|---------|--------|
| Consolidate rollback decision | heuristic (always) | data-driven (if $\Delta W_2 > \tau$) |
| Field stability tracking | N/A | time-series of $W_2$ distances |

**Risks:**
- $O(N^3)$ for exact assignment; sinkhorn needs careful tuning of $\varepsilon$.
- More of an observability feature than a core algorithmic improvement.

---

### P3.2 Persistent Homology (TDA)
**Owner:** TBD | **Estimate:** 5 days | **Status:** 🔴 Not started

**Problem:** `TDAMonitor` is a stub. No topological insight into memory structure.

**Mathematics:**
- Vietoris-Rips filtration on latent positions
- Persistent barcode: $H_0$ (clusters), $H_1$ (cycles/associative loops), $H_2$ (voids)
- Persistence entropy: $E = -\sum p_i \log p_i$ where $p_i = (d_i - b_i) / \sum (d_j - b_j)$

**Expected Impact:**
| Metric | Current | Target |
|--------|---------|--------|
| Cognitive complexity score | N/A | single number $E \in [0, \log N]$ |
| Associative loop detection | manual | automatic (non-trivial $H_1$ classes) |

**Risks:**
- Needs `ripser` or `gudhi` dependency; may not install cleanly on Windows.
- $O(N^2 \cdot 2^d)$ worst-case; only feasible for $N < 2000$ without approximation.
- Interesting research-wise, low production value.

---

## Summary Matrix

| ID | Feature | Impact | Complexity | Risk | Effort | Priority |
|----|---------|--------|------------|------|--------|----------|
| P0.1 | Riemannian SGD | 🔴 High | Medium | Medium | 3d | **DO FIRST** |
| P1.1 | Conformal Prediction | 🔴 High | Medium | Medium | 3d | **P1** |
| P1.2 | Local Bandwidth | 🟡 Medium | Low | Low | 1d | **P1** |
| P2.1 | Spectral Laplacian | 🟡 Medium | Medium | Medium | 2d | P2 |
| P2.2 | Kalman Filter | 🟡 Medium | High | High | 4d | P2 |
| P3.1 | Wasserstein | 🟢 Low | Medium | Low | 2d | P3 |
| P3.2 | Persistent Homology | 🟢 Low | High | High | 5d | P3 |

## Итоги трека P0–P2 (Май 2026)

Все 5 фич реализованы, протестированы (203 теста), интегрированы в `RTMDKField`:

| ID | Feature | Тесты | Статус | Файлы |
|----|---------|------|--------|-------|
| P0.1 | Riemannian SGD | 12 | ✅ | `geometry.py`, `core.py` |
| P1.1 | Conformal Prediction | 10 | ✅ | `memory/conformal.py`, `core.py` |
| P1.2 | Local Bandwidth | 10 | ✅ | `core.py` |
| P2.1 | Spectral Laplacian | 13 | ✅ | `memory/spectral.py`, `core.py` |
| P2.2 | Kalman Filter | 15 | ✅ | `memory/kalman.py`, `core.py`, `nodes.py` |

**Ключевые интеграционные точки в `core.py`:**
- `_build_node_cache()` → `adaptive_bandwidth` вычисляет `_cached_bw`
- `query()` → `_apply_conformal_filter()` + Kalman `uncertainty_weight()`
- `consolidate()` → `spectral_cluster_nodes()` перед greedy merge + `poincare_midpoint()` при `hyperbolic=True`
- `_do_merge()` → `kf.merge_covariance()` при `enable_kalman_filter=True`
- `add_node()` → инициализация `node.covariance` при `enable_kalman_filter=True`

## SOT Enhancement Track (SOT v2) — Май 2026

### SOT-A: Warm-Start PMI + IDF
**Status:** ✅ Done | **Impact:** Similarity related texts 0.74→0.91

**What:** Compute byte-bigram PMI + IDF from corpus at init; initialize embeddings via SVD of co-occurrence matrix.

---

### SOT-B: Engram Manager in RTMDKField
**Status:** ✅ Done | **Impact:** Direct field usage now supports engrams

**What:** Moved `engram_manager` initialization from `RTMDKMemory` into `RTMDKField.__init__`.

---

### SOT-C: Co-occurrence Dict Size Limit
**Status:** ✅ Done | **Impact:** OOM risk eliminated

**Problem:** `cooccurrence` dict grows without bound. Long-running instances will OOM.

**Solution:** `CooccurrenceStore` class with configurable `max_size` (default 100K). When threshold exceeded, drops lowest-weight entries via sort + truncate.

**Config:** `sot_max_cooccurrence: int = 100_000`

**Usage:**
```python
tok = SOTokenizer(latent_dim=64, max_cooccurrence=50_000)
tok.record_cooccurrence(tokens)  # auto-prunes when needed
stats = tok.cooccurrence.get_stats()
# {'size': 50000, 'max_size': 50000, 'total_inserts': 123456, 'total_prunes': 73456}
```

---

### SOT-D: Gradient Clipping in Feedback Loop
**Status:** ✅ Done | **Impact:** Prevents NaN projection updates

**What:** Added `np.clip(delta, -0.1, 0.1)` in `_sot_retrieval_feedback` projection update.

---

### SOT-E: Subword Seed (Preset Merges)
**Status:** ✅ Done | **Impact:** Vocab 256→353, encoding shorter by ~15%

**What:** Pre-seed with 500 common English byte bigrams + 200 trigrams.

---

### SOT-F: Attention Pooling
**Status:** ✅ Done | **Impact:** First/last token weighted; IDF-aware

**What:** `embed()` uses IDF weights + position bonus (first ×1.5, last ×1.2) instead of mean.

---

### SOT-G: Hard Negatives
**Status:** ✅ Done | **Impact:** Contrastive learning uses closest non-positives

**What:** `update_with_hard_negatives()` selects negatives by similarity rather than random.

---

### SOT-H: Retrieval Feedback
**Status:** ✅ Done | **Impact:** Embeddings adapt from query results

**What:** `_sot_retrieval_feedback()` updates token embeddings + projection using top/bottom results.

---

### SOT-I: Skip-gram Co-occurrence
**Status:** ✅ Done | **Impact:** Captures long-range associations

**What:** `skipgram_window` (default 3) records co-occurrence beyond adjacent tokens.

---

### SOT-J: SBERT Bootstrap + Word-Level Tokenization
**Status:** ✅ Done | **Impact:** Recall@1 0.22 → 0.80 (1000 QA)

**Problem:** Byte-level tokens cannot capture word-level semantics. "earthquake" and "quake" share no bytes → SBERT bootstrap on bytes gives marginal gain (0.23 vs 0.22).

**Solution:** 
1. **Word-level tokenization** (`tokenization_mode="word"`): split on whitespace/punctuation, each word gets an embedding.
2. **SBERT bootstrap** (`bootstrap_from_teacher`): iterative update of word embeddings toward teacher targets, then ridge regression on projection matrix.

**Benchmark (200 QA):**
| Method | Recall@1 |
|--------|----------|
| BM25 | 0.730 |
| Byte v1 | 0.300 |
| Word v1 | 0.360 |
| Word + warm-start | 0.410 |
| **Word + SBERT bootstrap** | **0.670** |

**Benchmark (1000 QA):**
| Method | Recall@1 |
|--------|----------|
| BM25 | 0.691 |
| Byte v1 | 0.224 |
| Word v1 | 0.356 |
| **Word + SBERT bootstrap** | **0.799** |

**Analysis:**
- Word-level solves the fundamental byte-token mismatch.
- SBERT bootstrap gives semantic cold-start: word embeddings initialized from teacher signals.
- **0.799 > BM25 0.691**: SOT now exceeds BM25 on this benchmark.

**Config:**
```python
RTMDKConfig(
    latent_dim=64,
    sot_enabled=True,
    sot_tokenization_mode="word",
)
# Then call field.sot_tokenizer.bootstrap_from_teacher(texts, teacher_fn)
```

---

### SOT-K: FastText Bootstrap (Lightweight Alternative)
**Status:** ✅ Done | **Impact:** Recall@1 0.769 (1000 QA), bootstrap time 0.23s vs 12s SBERT

**Problem:** SBERT requires torch+transformers+sentence-transformers (~2GB deps, 80MB model). Too heavy for edge deployment.

**Solution:** Use gensim KeyedVectors (GloVe/FastText) as teacher. 128MB model, no torch dependency, 0.23s bootstrap.

**Benchmark (1000 QA):**
| Method | Recall@1 | Bootstrap Time |
|--------|----------|----------------|
| BM25 | 0.691 | — |
| Word + SBERT | 0.799 | 12s |
| **Word + FastText** | **0.769** | **0.23s** |

**Analysis:**
- FastText gives 96% of SBERT quality with 2% of bootstrap time.
- No torch dependency — pure numpy + gensim.
- Model is 128MB (glove-wiki-gigaword-100), but can be smaller with glove-twitter-25 (~100MB, 25d).

**Usage:**
```bash
# Download model once
gensim.downloader api.load('glove-wiki-gigaword-100') → save

# CLI bootstrap
python -m rtmdk bootstrap-fasttext model.model corpus.json -o state.json

# Or auto-bootstrap
RTMDKConfig(
    latent_dim=64,
    sot_enabled=True,
    sot_tokenization_mode="word",
    sot_bootstrap_corpus="corpus.json",
    sot_bootstrap_fasttext_model="glove-wiki-gigaword-100.model",
)
```

---

## System Risk Audit (Май 2026)

### 🔴 Critical

| Risk | Impact | Mitigation |
|------|--------|------------|
| SOT cold start ≈ 30% recall | Semantic retrieval weak | Word-level + SBERT bootstrap ✅ |
| Byte-level tokenization | "not good" == "good not" | Word-level ✅ |
| Co-occurrence unbounded growth | OOM after months | CooccurrenceStore ✅ (SOT-C) |

### 🟡 High

| Risk | Impact | Mitigation |
|------|--------|------------|
| Conformal prediction 300ms latency | UX degradation | Reduce min_calib; cache threshold |
| Spectral timeout 500ms | Incomplete clustering | Adaptive timeout based on N |
| Kalman +50% query latency | Slow retrieval | Diagonal approx ✅; disable by default |
| HNSW phantom nodes after delete | Returns deleted nodes | Rebuild HNSW after consolidate |

### 🟢 Medium

| Risk | Impact | Mitigation |
|------|--------|------------|
| 222 tests only | Missed edge cases | Add concurrency + memory leak tests |
| Windows-only CI | Linux/macOS bugs | Add cross-platform CI |
| Config drift (100+ params) | Unexpected interactions | Config validation matrix |

---

**Recommended roadmap (completed):**
1. ✅ **Day 1:** SOT-J (SBERT bootstrap on full corpus) + benchmark at 1000 QA
2. ✅ **Day 2:** SOT-C (co-occurrence limit)
3. ✅ **Day 3:** Integration tests + word vocab pruning
4. ✅ **Day 4:** FastText bootstrap (SOT-K) + CLI + docs

**Next steps:**
1. Memory leak profiling under long-running load
2. Cross-platform CI (Linux/macOS)
3. Docker image with pre-built FastText model
4. ✅ Benchmark automation script (`benchmark.py`) — supports bm25, byte, word, word_fasttext, word_sbert
