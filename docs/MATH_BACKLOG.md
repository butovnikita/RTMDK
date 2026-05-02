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
**Owner:** TBD | **Estimate:** 3 days | **Status:** 🔴 Not started

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
**Owner:** TBD | **Estimate:** 1 day | **Status:** 🔴 Not started

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

**Acceptance Criteria:**
- [ ] `_build_node_cache()` precomputes `kdist` array
- [ ] `_compute_resonance_chunk()` uses per-node `bw` vector
- [ ] No regression on `test_chunked_query_matches_non_chunked`

---

## P2 — Medium Value (observability & stability)

### P2.1 Spectral Graph Laplacian for Consolidation
**Owner:** TBD | **Estimate:** 2 days | **Status:** 🔴 Not started

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

**Acceptance Criteria:**
- [ ] Fallback to greedy merge if eigendecomposition fails or timeout >500ms
- [ ] `test_domain_memory` backward compatibility preserved
- [ ] Eigengap auto-$k$ capped at `max_consolidation_clusters` config

---

### P2.2 Kalman Filtering on Manifold (Riemannian EKF)
**Owner:** TBD | **Estimate:** 4 days | **Status:** 🔴 Not started

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

**Acceptance Criteria:**
- [ ] Optional feature: `enable_kalman_filter: bool = False`
- [ ] Uncertainty used to weight retrieval: `score *= 1 / (1 + \text{tr}(\Sigma))`
- [ ] Memory overhead capped: store diagonal only if `kalman_diagonal_approx: bool = True`

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

**Recommended roadmap:**
1. **Week 1:** P0.1 (Riemannian SGD) — фундаментальный фикс
2. **Week 2:** P1.2 (Local bandwidth) — быстрый win + P1.1 (Conformal) — production confidence
3. **Week 3–4:** P2.1 (Spectral) — если consolidate всё ещё проблематичен
4. **Month 2+:** P2.2 / P3.x — research tier, по необходимости
