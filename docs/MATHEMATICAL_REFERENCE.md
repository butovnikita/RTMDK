# RTMDK Mathematical Reference

> Comprehensive compilation of all formulas, theorems, proofs, algorithms, complexity analyses, and optimization methods found across the RTMDK codebase (v8.3.0).

---

## Table of Contents

1. [Notation & Primitives](#1-notation--primitives)
2. [SOT v2.0 Embedder Mathematics](#2-sot-v20-embedder-mathematics)
3. [Field Resonance Memory Mathematics](#3-field-resonance-memory-mathematics)
4. [Geometry & Manifolds](#4-geometry--manifolds)
5. [Graph & Spectral Methods](#5-graph--spectral-methods)
6. [Retrieval & Routing Mathematics](#6-retrieval--routing-mathematics)
7. [Conformal Prediction](#7-conformal-prediction)
8. [State Space Model & ODE Dynamics](#8-state-space-model--ode-dynamics)
9. [Pipeline & Production Mathematics](#9-pipeline--production-mathematics)
10. [Complexity Analysis](#10-complexity-analysis)
11. [Performance Benchmarks](#11-performance-benchmarks)
12. [Theorems & Proofs](#12-theorems--proofs)
- [Appendix A: Configuration Presets](#appendix-a-configuration-presets-with-mathematical-parameters)
- [Appendix B: Parameter Quick Reference](#appendix-b-key-parameter-quick-reference-table)
- [Appendix C: Optimization Summary](#appendix-c-optimization-summary-table)

---

## 1. Notation & Primitives

### 1.1 Memory Node

Each memory node is a tuple:

$$M_i = (L_i, \varphi_i, A_i, S_i, \tau_i, C_i, T_i, E_i, \Gamma_i)$$

| Symbol | Domain | Meaning |
|--------|--------|---------|
| $L_i \in \mathbb{R}^d$ | Latent space | Projected position (default $d = 256$) |
| $\varphi_i \in [0, 2\pi)$ | Phase | Oscillator coordinate |
| $A_i \in [0, 1]$ | Amplitude | Signal strength |
| $S_i \in [0, 1]$ | Salience | Importance / relevance |
| $\tau_i \in \mathbb{R}_{\geq 0}$ | Tension | Neighborhood heterogeneity |
| $C_i \in \mathbb{R}^D$ | Embedding | Original dense embedding (default $D = 768$) |
| $T_i \in \mathbb{R}$ | Timestamp | Creation time |
| $E_i$ | Set | Engrams containing this node |
| $\Gamma_i$ | Set | Causal edges |

### 1.2 Latent Projection

$$L_i = P \cdot (C_i - \mu)$$

where $P \in \mathbb{R}^{d \times D}$ is the projection matrix and $\mu \in \mathbb{R}^D$ is the mean vector. Implemented via incremental PCA (`rtmdk/support/projection.py`).

### 1.3 Common Symbols

| Symbol | Default | Meaning | Location |
|--------|---------|---------|----------|
| $\beta$ | 1.0 | Bandwidth (spatial kernel width) | `config.py:445` |
| $\kappa$ | 0.3 | Phase coupling strength | `config.py:444` |
| $\rho$ | 0.997 | Decay rate | `config.py:462` |
| $\gamma$ | 0.15 | Gate temperature | `config.py:498` |
| $\tau^*$ | 0.15 | Tension threshold | `config.py:464` |
| $\sigma$ | — | Spectral clustering bandwidth | `spectral.py` |
| $R$ | 0.85 | Poincaré ball radius | `config.py:642` |
| $\lambda$ | 1.0 | Ridge regularization | `self_organizing_field.py` |
| $\alpha$ | 1.0 | Hybrid blending weight | `config.py:489` |
| $k_1$ | 1.5 | BM25 term frequency saturation | `config.py:484` |
| $b$ | 0.75 | BM25 length normalization | `config.py:485` |

---

## 2. SOT v2.0 Embedder Mathematics

> Source files: `rtmdk/memory/sot_v2/`, `rtmdk/memory/self_organizing_field.py`, `rtmdk/memory/engram_cache.py`

### 2.1 MI-Subword Tokenization

**Mutual Information** between adjacent tokens $a$ and $b$:

$$\text{MI}(a, b) = \log \frac{P(a, b)}{P(a) \, P(b)}$$

**Theorem 1** (Gage 1994; Sennrich et al. 2016). Greedy merge maximizing MI yields minimum cross-entropy tokenization.

Cross-entropy change after merge:

$$\Delta L = -\bigl[ |C|_{ab} \log P(c) - |C|_a \log P(a) - |C|_b \log P(b) \bigr]$$

### 2.2 Pointwise Mutual Information (PMI) and SPPMI

**PMI** between vocabulary items $i$ and $j$:

$$\text{PMI}(i, j) = \log \frac{\#(i, j) \cdot |C|}{\#(i) \cdot \#(j)}$$

**Shifted Positive PMI (SPPMI)**:

$$w_{ij} = \max\bigl(0, \; \text{PMI}(i, j) - \log k\bigr)$$

For $k = 1$ this reduces to standard PPMI.

### 2.3 Spectral Embedding & Truncated SVD

**Normalized Graph Laplacian**:

$$L_{\text{sym}} = I - D^{-1/2} A \, D^{-1/2}$$

where $A$ is the affinity matrix with $A_{ij} = w_{ij}$ and $D_{ii} = \sum_j w_{ij}$.

**Truncated SVD** of the PMI matrix:

$$\text{PMI} = U \, \Sigma \, V^T \quad \Rightarrow \quad x_i = U_i \cdot \sqrt{\Sigma_i}$$

**Randomized SVD** (Halko et al., 2011) for large matrices:
- Complexity: $O(nd^2)$ instead of $O(n^3)$ for $n > 2000$
- Used automatically when `n_valid > 2000`

**Dense vs Sparse PMI Path**:
- Vocabulary $\leq 5000$: dense `np.linalg.svd` (faster)
- Vocabulary $> 5000$: `scipy.sparse` + `TruncatedSVD` (memory-safe)

### 2.4 Smooth Inverse Frequency (SIF)

**Theorem 2** (Arora et al. 2017). Under the random-walk discourse model, the Maximum Likelihood Estimator (MLE) for sentence embedding is the SIF weighted average:

$$P(w \mid c_t) \propto \exp(\langle v_w, c_t \rangle)$$

$$v_s = \frac{1}{|s|} \sum_{w \in s} \frac{a}{a + p(w)} \cdot v_w$$

where:
- $v_w$ is the word embedding
- $p(w)$ is the unigram probability
- $a$ is a smoothing parameter (default $a = 0.001$)

**Adaptive SIF parameter**:

$$a^* = P_{10}\bigl(\{ p(w) \}_{w \in V}\bigr)$$

where $P_{10}$ is the 10th percentile of unigram probabilities.

### 2.5 Principal Component Removal

After computing the SIF embedding, remove the first principal component to debias:

$$v_s^{\text{corrected}} = v_s - \langle v_s, u_1 \rangle \, u_1$$

where $u_1$ is the dominant eigenvector of the covariance matrix (computed via power iteration for efficiency).

### 2.6 InfoNCE Contrastive Loss

For contrastive fine-tuning of the SOT v2 embedder:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\Bigl[\log \frac{\exp(\text{sim}(q, p^+) / \tau)}{\sum_{i} \exp(\text{sim}(q, p_i) / \tau)}\Bigr]$$

**Logit shifting** for numerical stability:

$$\log \sum_i \exp(x_i) = \max_j(x_j) + \log \sum_i \exp(x_i - \max_j(x_j))$$

### 2.7 Orthogonal Procrustes Alignment

**Theorem 3** (Schönemann 1966). Given $X, Y \in \mathbb{R}^{n \times d}$, find orthogonal $R$ minimizing $\|XR - Y\|_F^2$:

$$R = U \, V^T \quad \text{where} \quad U \, \Sigma \, V^T = \text{SVD}(X^T Y)$$

Used for aligning SOT v2 embeddings with teacher model (e.g., SBERT) embeddings.

### 2.8 Contrastive Distillation (Teacher-Student)

$$\mathcal{L}_{\text{distill}} = \alpha \cdot \mathcal{L}_{\text{MSE}}(W \cdot v_{\text{student}}, v_{\text{teacher}}) + (1 - \alpha) \cdot \mathcal{L}_{\text{InfoNCE}}$$

Student projection matrix $W$ is orthonormalized after each update to prevent embedding collapse:

$$W \leftarrow W \cdot (W^T W)^{-1/2}$$

### 2.9 Online Matrix Factorization

For incremental vocabulary updates:

$$W_{t+1} = W_t - \eta \cdot \nabla_{W} \|X - W_t H_t\|_F^2$$

$$H_{t+1} = H_t - \eta \cdot \nabla_{H} \|X - W_t H_t\|_F^2$$

### 2.10 Quantum Resonance Retrieval

**Density matrix** representation of the document corpus:

$$\rho = \sum_i p_i \, |\psi_i\rangle\langle\psi_i|$$

$$
\rho_d = \frac{1}{N} \sum_{i=1}^{N} |v_i\rangle\langle v_i| + \varepsilon I
$$

**Query resonance score** (low-rank factorization for speed):

$$S(q, \rho_d) = \langle q | \rho_d | q \rangle = q^T \rho_d \, q = \frac{1}{N} \sum_{i=1}^{N} (q \cdot v_i)^2 + \varepsilon \|q\|^2 = \|V^T q\|^2 + \varepsilon \|q\|^2$$

where $V^T q$ is a fast matrix-vector product.

### 2.11 MaxSim Operator

For late-interaction retrieval:

$$\text{MaxSim}(q, d) = \sum_{i \in q} \max_{j \in d} \, \text{sim}(e_i^{(q)}, e_j^{(d)})$$

### 2.12 Hybrid BM25+SIF Retrieval

**Normalized hybrid score**:

$$S_{\text{hybrid}}(q, d) = \alpha \cdot \tilde{S}_{\text{dense}} + (1 - \alpha) \cdot \tilde{S}_{\text{sparse}}$$

where $\tilde{S}$ denotes min-max normalized scores.

**Theorem 4** (Clinchant & Gaussier 2010). If errors are conditionally independent with equal variance, the optimal fusion weight is $\alpha = 0.5$.

### 2.13 FastText OOV Character N-Grams

For out-of-vocabulary words, character n-grams with boundary markers `#word#`:

$$e_{\text{word}} = \frac{\displaystyle\sum_{g \in \text{ngrams}(w)} \frac{a}{a + p(g)} \cdot e_g}{\displaystyle\sum_{g} \frac{a}{a + p(g)}}$$

where $e_g$ is the mean of word embeddings for all corpus words containing n-gram $g$.

### 2.14 SOT Contrastive Learning in Self-Organizing Field

Adaptive learning rate scaling:

$$\text{if } d_{\text{token}} > d_{\text{latent}}: \quad \eta_{\text{eff}} = \eta \cdot \sqrt{\frac{d_{\text{token}}}{d_{\text{latent}}}}$$

Token update rule:

$$\Delta = \eta \cdot (\mu_{\text{pos}} - \text{emb}) - \eta \cdot 0.1 \cdot \sum_{k} (\mu_{\text{neg}_k} - \text{emb})$$

PMI-based warm start:

$$\text{pmi} = \log\bigl(\frac{p_{ab}}{p_a \, p_b} + 10^{-10}\bigr)$$

$$\text{if pmi} > 0: \quad \text{token\_embeddings}[a] \mathrel{+}= 0.05 \cdot \text{pmi} \cdot (\text{emb}[b] - \text{emb}[a])$$

---

## 3. Field Resonance Memory Mathematics

> Source files: `rtmdk/memory/resonance.py`, `rtmdk/memory/field.py`, `rtmdk/memory/topology_manager.py`, `rtmdk/memory/scheduler.py`, `rtmdk/memory/utils.py`

### 3.1 Resonance Response (Single Node)

**Distance computation** (Euclidean or hyperbolic):

$$d_i = \|L_q - L_i\|_2 \quad \text{(Euclidean)}$$

$$d_{\text{Poincaré}} = R \cdot \text{arccosh}\Bigl(1 + \frac{2 \|u - v\|^2}{(R^2 - \|u\|^2)(R^2 - \|v\|^2)/R^2}\Bigr) \quad \text{(Hyperbolic)}$$

**Spatial kernel** (configurable):

$$K_{\text{spatial}} = \begin{cases}
\exp\bigl(-d_i^2 / (2\beta^2)\bigr) & \text{Gaussian} \\[6pt]
0.5 + 0.5 \cdot \dfrac{\langle q, n_i \rangle}{\|q\| \cdot \|n_i\|} & \text{Cosine} \\[6pt]
\exp(-d_i / \beta) & \text{Laplacian}
\end{cases}$$

**Phase alignment**:

$$\Delta\varphi_i = \varphi_i - \varphi_q$$

$$K_{\text{phase}} = 0.5 + 0.5 \cdot \cos(\Delta\varphi_i)$$

**Full resonance response**:

$$R_i = K_{\text{spatial}} \cdot \bigl[(1 - \kappa) + \kappa \cdot K_{\text{phase}}\bigr] \cdot A_i \cdot S_i$$

**Modulation**:

$$\text{gate} = \sigma\bigl((\tau_i - \tau^*) / \gamma\bigr) \quad \text{if soft gates enabled}$$

$$\text{causal\_boost} = 1.0 + 0.1 \cdot \sum_{p \in \text{parents}} \text{causal\_strength}[p]$$

$$R_i^{\text{final}} = R_i \cdot \text{gate} \cdot w_{\text{modal}} \cdot \text{causal\_boost}$$

### 3.2 Batch Resonance (Vectorized)

$$D = \text{cdist}(Q_{\text{latents}}, P_{\text{positions}}) \in \mathbb{R}^{n_q \times n_n}$$

$$\text{spatial} = \exp(-D^2 / (2\beta^2))$$

$$\Delta\Phi = \varphi_q^{\top} \mathbf{1}^T - \mathbf{1} \, \varphi_n^T$$

$$\text{phase\_align} = 0.5 + 0.5 \cdot \cos(\Delta\Phi)$$

$$\text{response} = \text{spatial} \cdot \bigl((1 - \kappa) + \kappa \cdot \text{phase\_align}\bigr) \cdot A^T \cdot S^T$$

### 3.3 Int8-Quantized Fast Path

For accelerated resonance computation with quantized embeddings:

$$\|q - p_{\text{deq}}\|^2 = \|q\|^2 + \|p_{\text{deq}}\|^2 - 2 \, q \cdot p_{\text{deq}}$$

where $p_{\text{deq}} = \text{int8\_p} \cdot \text{scale\_per\_vector}$ and:

$$\text{dot} = (q \cdot \text{int8\_p}^T) \cdot \text{scale\_per\_vector}$$

### 3.4 Adaptive Phase Coupling

Calibrated on `comprehensive_500` and `qa_1000_en` datasets:

$$\text{gap} = \text{cos\_sim}(q, \text{top}_1) - \text{cos\_sim}(q, \text{top}_2)$$

$$\kappa = \begin{cases}
0.0 & \text{if top-1 accuracy} \geq 0.95 \text{ (unambiguous)} \\[4pt]
0.1 & \text{if top-1 accuracy} \geq 0.5 \text{ (moderate)} \\[4pt]
0.15 & \text{otherwise}
\end{cases}$$

Source: `rtmdk/memory/adaptive_pc.py`

### 3.5 Amplitude, Salience, and Soft Gating

**Soft gate** (sigmoid):

$$g_i = \sigma\Bigl(\frac{\tau_i - \tau^*}{\gamma}\Bigr) = \frac{1}{1 + \exp\bigl(-(\tau_i - \tau^*) / \gamma\bigr)}$$

Default $\gamma = 0.15$.

### 3.6 Tension Computation

Over $k$-nearest neighbors within `neighborhood_radius` (default 2.0):

$$\tau_i = 0.6 \cdot \bigl[\text{std}(\cos(\varphi_{\text{neighbors}})) + \text{std}(\sin(\varphi_{\text{neighbors}}))\bigr] + 0.4 \cdot \text{std}(S_{\text{neighbors}})$$

If fewer than 2 neighbors: $\tau_i = 0.0$.

### 3.7 Exponential Decay & Half-Life

$$S_i \leftarrow S_i \cdot \rho \qquad A_i \leftarrow A_i \cdot \rho$$

**Half-life**:

$$t_{1/2} = \frac{\ln(0.5)}{\ln(\rho)}$$

At $\rho = 0.997$: $t_{1/2} \approx 230$ steps.  
At $\rho = 0.999$: $t_{1/2} \approx 693$ steps.

### 3.8 Tier-Specific & Adaptive Decay

**Tier decay rates** (`config.py`):

| Tier | Decay Rate | Half-life |
|------|-----------|-----------|
| Episodic | 0.992 | ~86 steps |
| Semantic | 0.999 | ~693 steps |
| Procedural | 1.0 | $\infty$ (no decay) |

**Adaptive decay** (feedback-driven, `advanced_retrieval.py`):

$$\delta = (0.5 - \text{quality}) \cdot 0.01$$

$$\rho_{\text{new}} = \text{clip}(\rho_{\text{current}} + \delta, \; \rho_{\min}, \; \rho_{\max})$$

- Quality $> 0.5$ → slower decay (increase half-life)
- Quality $< 0.5$ → faster decay (decrease half-life)

### 3.9 Cross-Modal Resonance

**Modal coupling** between query and node phases:

$$\text{modal\_coupling} = \cos(\varphi_q - \varphi_n)$$

$$\text{boost} = 1.0 + w_{\text{cross\_modal}} \cdot \text{modal\_coupling}$$

**Default phase offsets** by modality (`config.py`):

| Modality | Phase Offset |
|----------|-------------|
| Text | $0$ |
| Code | $\pi/4$ |
| Audio | $\pi/2$ |
| Vision | $3\pi/4$ |
| Metrics | $\pi$ |

### 3.10 Attention Bias & Session Boosting

$$\text{score} \mathrel{*}= (1.0 + 0.2 \cdot \text{causal\_boost})$$

$$\text{score} \mathrel{*}= \max(0.5, \; 1.0 - \tau_i)$$

$$\text{score} \mathrel{*}= (1.0 + 0.3 \cdot \text{goal\_relevance})$$

**Session match boost** (`core.py`):

$$\text{boost}_{\text{session}} = 1.5 \quad \text{(50\% boost when session IDs match)}$$

### 3.11 Causal Traversal Scoring

**BFS hop score**:

$$\text{Score}_{\text{hop}} = \text{Score}_{\text{parent}} \cdot \text{decay}^{\text{depth}} \cdot \text{causal\_strength}$$

**Combined score**:

$$\text{Score}_{\text{new}} = \begin{cases}
\text{Score}_{\text{resonance}} + 0.3 \cdot \text{Score}_{\text{causal}} & \text{for existing nodes} \\[4pt]
\text{Score}_{\text{causal}} \cdot 0.7 & \text{for newly discovered nodes}
\end{cases}$$

### 3.12 Engram Model

**Engram definition**:

$$G_j = \{(n_1, w_1), (n_2, w_2), \ldots, (n_k, w_k)\}, \qquad \sum w_m = 1, \; 2 \leq k \leq 20$$

**Engram centroid**:

$$\text{CG}_j = \sum_m w_m \cdot C_{n_m}$$

**Engram strength**:

$$\text{Strength}(G_j) = \prod_m \bigl(1 + \text{activation\_count} / 100\bigr) \cdot \exp(-\text{age} / t_{1/2})$$

Default half-life = 346 steps (at `engram_decay_rate = 0.998`).

### 3.13 Dialectical Consolidation

**Merge rules** when two nodes are consolidated:

$$L_{\text{new}} = 0.5 \cdot (L_i + L_j)$$

$$\varphi_{\text{new}} = \text{atan2}\bigl(0.5 \sin(\varphi_i) + 0.5 \sin(\varphi_j), \; 0.5 \cos(\varphi_i) + 0.5 \cos(\varphi_j)\bigr)$$

$$A_{\text{new}} = \min\bigl(1.0, \; 0.8 \cdot (A_i + A_j)\bigr)$$

$$S_{\text{new}} = 0.7 \cdot (S_i + S_j)$$

### 3.14 Ridge Regression for Projection

In `self_organizing_field.py`:

$$W = (X^T X + \lambda I)^{-1} X^T Y, \qquad \lambda = 1.0$$

### 3.15 Chi-Squared Critical Values

In `core.py`:

$$\chi^2_{\text{crit}, \text{df}=1} = 3.84 \quad (p = 0.05)$$

$$\chi^2_{\text{crit}, \text{df}=2} = 5.99 \quad (p = 0.05)$$

---

## 4. Geometry & Manifolds

> Source files: `rtmdk/memory/geometry.py`, `rtmdk/memory/kalman.py`, `rtmdk/memory/field.py`

### 4.1 Hyperbolic Poincaré Ball

All formulas are correct for arbitrary ball radius $R$ (not just $R = 1$).

**Poincaré distance**:

$$\delta = \|u - v\|^2$$

$$\text{denom} = \frac{(R^2 - \|u\|^2)(R^2 - \|v\|^2)}{R^2}$$

$$d_{\text{Poincaré}}(u, v) = R \cdot \text{arccosh}\Bigl(1 + \frac{2\delta}{\text{denom}}\Bigr)$$

**Möbius addition**:

$$\text{num} = \Bigl(1 + \frac{2 \langle x, y \rangle}{R^2} + \frac{\|y\|^2}{R^2}\Bigr) x + \Bigl(1 - \frac{\|x\|^2}{R^2}\Bigr) y$$

$$\text{den} = 1 + \frac{2 \langle x, y \rangle}{R^2} + \frac{\|x\|^2 \|y\|^2}{R^4}$$

$$x \oplus y = \frac{\text{num}}{\text{den}}$$

**Exponential map**:

$$\lambda_{\text{base}} = \frac{2}{1 - \|\text{base}\|^2 / R^2}$$

$$c = R \cdot \frac{\tanh\bigl(\lambda_{\text{base}} \cdot \|\text{tangent}\| / (2R)\bigr)}{\|\text{tangent}\|}$$

$$\exp_{\text{base}}(\text{tangent}) = \text{base} \oplus (c \cdot \text{tangent})$$

**Logarithmic map** (inverse of exponential):

$$\text{Implemented in } \log\_\text{map\_poincare}(z, x, R)$$

### 4.2 Riemannian SGD on Poincaré Ball

**Riemannian gradient** from Euclidean gradient:

$$\lambda_x = \frac{2}{1 - \|x\|^2 / R^2}$$

$$\nabla_R f(x) = \frac{1}{\lambda_x^2} \, \nabla_E f(x)$$

**SGD step**:

$$x_{\text{new}} = \exp_x(-\eta \cdot \nabla_R f(x))$$

**Poincaré midpoint**:

$$m = \exp_a\Bigl(\frac{1}{2} \log_a(b)\Bigr)$$

Implementation in `field.py` (lines 1033–1044):

$$\text{grad}_E = \text{target} - \text{node.latent\_pos}$$

$$\text{norm\_sq} = \|\text{node.latent\_pos}\|^2$$

$$\text{conformal} = \frac{(1 - \text{norm\_sq}/R^2)^2}{4}$$

$$\text{grad}_R = \text{conformal} \cdot \text{grad}_E$$

$$\text{node.latent\_pos} = \exp_{\text{Poincaré}}(-\text{lr} \cdot \text{grad}_R, \; \text{node.latent\_pos}, \; R)$$

### 4.3 Riemannian EKF (Kalman Filter)

> Source: `rtmdk/memory/kalman.py`

**Prediction** (add process noise):

$$\Sigma_i \leftarrow \Sigma_i + Q \cdot I$$

**Innovation** (hyperbolic mode):

$$y = \log_{L_i}(z) \quad \text{(tangent space innovation)}$$

**Kalman gain** (diagonal approximation):

$$S = \text{cov} + R$$

$$K = \frac{\text{cov}}{S + 10^{-10}}$$

**Update**:

$$L_i \leftarrow \exp_{L_i}(K \cdot y) \quad \text{(hyperbolic)}$$

$$L_i \leftarrow L_i + K \cdot y \quad \text{(Euclidean)}$$

$$\Sigma_i \leftarrow (1 - K) \cdot \text{cov}$$

**Full matrix update**:

$$K = \Sigma \, H^T (H \, \Sigma \, H^T + R)^{-1}$$

$$x_{t|t} = x_{t|t-1} + K \cdot (z_t - H \cdot x_{t|t-1})$$

$$\Sigma_{t|t} = (I - K \, H) \, \Sigma_{t|t-1}$$

**Merge information weighting**:

$$\Sigma_{\text{new}} = \bigl(w_a \, \Sigma_a^{-1} + w_b \, \Sigma_b^{-1}\bigr)^{-1}$$

**Retrieval weighting**:

$$\text{weight}_i = \frac{1}{1 + \text{tr}(\Sigma_i)}$$


## 5. Graph & Spectral Methods

> Source files: `rtmdk/memory/spectral.py`, `rtmdk/memory/topology_manager.py`, `rtmdk/memory/cpen_ode.py`

### 5.1 Spectral Clustering

**Affinity matrix**:

$$W_{ij} = \exp\Bigl(-\frac{\|L_i - L_j\|^2}{2\sigma^2}\Bigr) \cdot \frac{1 + \cos(\varphi_i - \varphi_j)}{2}$$

**Degree matrix**:

$$D_{ii} = \sum_j W_{ij}$$

**Normalized Laplacian**:

$$L_{\text{sym}} = I - D^{-1/2} W \, D^{-1/2}$$

**Spectral embedding**: bottom-$k$ eigenvectors of $L_{\text{sym}}$ → $k$-means clustering.

**Eigengap heuristic** for optimal $k$:

$$k^* = \arg\max_k (\lambda_{k+1} - \lambda_k)$$

### 5.2 PC-Algorithm for Causal Discovery

**Expected co-occurrence**:

$$E(A, B) = \frac{N(A) \cdot N(B)}{N_{\text{total}}}$$

**Chi-squared test**:

$$\chi^2 = \frac{\bigl(N(A, B) - E(A, B)\bigr)^2}{E(A, B)}$$

Edge removed if $\chi^2 < 3.84$ ($p > 0.05$, 1 degree of freedom).

### 5.3 ODE Coupling (Parent-Child Dynamics)

> Source: `rtmdk/memory/cpen_ode.py`

**Parent ODE** (slow latent dynamics):

$$\frac{dx}{dt} = -x + W \cdot \tanh(x) + I$$

where $x$ is the concatenated latent positions of all nodes, $W$ is a weight matrix, and $I$ is external input.

**Child ODE** (fast local amplitude/phase updates per node):

$$\frac{dA_i}{dt} = \eta \cdot (\text{input}_i \cdot A_i - \text{decay} \cdot A_i)$$

$$\frac{d\varphi_i}{dt} = \omega_i + \text{coupling} \cdot \sum_j \sin(\varphi_j - \varphi_i)$$

**Vectorized phase dynamics**:

$$\sum_j \sin(\varphi_j - \varphi_i) = \Bigl(\sum_j \sin(\varphi_j)\Bigr) \cos(\varphi_i) - \Bigl(\sum_j \cos(\varphi_j)\Bigr) \sin(\varphi_i)$$

---

## 6. Retrieval & Routing Mathematics

> Source files: `rtmdk/support/bm25.py`, `rtmdk/production/bm25_fallback.py`, `rtmdk/production/cascade_router.py`, `rtmdk/memory/routing_manager.py`, `rtmdk/support/hnsw.py`, `rtmdk/support/adaptive_bandwidth.py`, `rtmdk/production/advanced_retrieval.py`

### 6.1 BM25 (Robertson-Jones)

**Inverse document frequency**:

$$\text{idf} = \log\Bigl(\frac{N - \text{df} + 0.5}{\text{df} + 0.5} + 1.0\Bigr)$$

**Term frequency component**:

$$\text{numerator} = \text{tf} \cdot (k_1 + 1)$$

$$\text{denominator} = \text{tf} + k_1 \cdot \Bigl(1 - b + b \cdot \frac{\text{doc\_len}}{\text{avg\_doc\_len}}\Bigr)$$

$$\text{score} = \text{idf} \cdot \frac{\text{numerator}}{\text{denominator}}$$

Defaults: $k_1 = 1.5$, $b = 0.75$.

### 6.2 Hybrid Blending

$$\text{blended\_score} = \alpha \cdot \text{resonance\_score} + (1 - \alpha) \cdot \text{bm25\_score}$$

Default `hybrid_alpha = 1.0` (pure RTMDK).

**Advanced hybrid retrieval** (`advanced_retrieval.py`) with min-max normalization:

$$\tilde{s}_{\text{rtmdk}} = \frac{s_{\text{rtmdk}}}{\max(s_{\text{rtmdk}}) + 10^{-8}}$$

$$\tilde{s}_{\text{bm25}} = \frac{s_{\text{bm25}}}{\max(s_{\text{bm25}}) + 10^{-8}}$$

$$\tilde{s}_{\text{cosine}} = \frac{\cos\_\text{sim} + 1}{2} \in [0, 1]$$

$$S_{\text{combined}} = 0.40 \cdot \tilde{s}_{\text{rtmdk}} + 0.35 \cdot \tilde{s}_{\text{bm25}} + 0.25 \cdot \tilde{s}_{\text{cosine}}$$

### 6.3 Cascade Router

Keyword-based routing with regex pattern scoring (`cascade_router.py`):

$$\text{causal\_score} = \text{count}(\text{causal keywords})$$

$$\text{factual\_score} = \text{count}(\text{factual keywords})$$

$$\text{route} = \begin{cases}
\text{CAUSAL (deep pipeline)} & \text{if causal\_score} \geq 0.3 \\[4pt]
\text{FACTUAL (fast pipeline)} & \text{if factual\_score} \geq 0.3 \\[4pt]
\text{AMBIGUOUS (standard pipeline)} & \text{otherwise}
\end{cases}$$

### 6.4 Sparse Shard Routing

Shard center distances (`routing_manager.py`):

$$d_s = \|\text{shard\_center}_s - L_q\|$$

$$\text{shard\_router\_score}_s = \frac{1}{1 + d_s}$$

Return top-$s$ shards with highest scores. Shard centers updated via k-means on node latent positions.

### 6.5 Local Adaptive Bandwidth (k-NN KDE)

$$\beta_i = \beta_{\text{global}} \cdot \sqrt{\frac{d_k(i)}{d_{\text{med}}}}$$

$$K_{\text{spatial}}(i) = \exp\Bigl(-\frac{\|L_q - L_i\|^2}{\beta_i}\Bigr)$$

where $d_k(i)$ is the distance to the $k$-th nearest neighbor and $d_{\text{med}}$ is the median of all $d_k$.

### 6.6 Adaptive Bandwidth Optimizer

Random-search bandwidth optimization (`support/adaptive_bandwidth.py`):

**Candidate sampling** (log-uniform):

$$\beta \sim \exp\bigl(\mathcal{U}(\ln 0.1, \; \ln 10.0)\bigr) \cdot \sqrt{d}$$

**Objective**: maximize Recall@K on synthetic probes (self-retrieval rate).

**Evaluation**:

$$\text{score}(\beta) = \frac{1}{|\text{probes}|} \sum_{i \in \text{probes}} \mathbb{1}[i \in \text{top\_k}(\beta)]$$

### 6.7 Naive Graph Index (HNSW Alias)

> Note: `rtmdk/support/hnsw.py` implements a flat greedy graph index, NOT a true HNSW.

- `insert`: $O(N)$ — scans all existing nodes
- `search`: $O(N)$ worst case — greedy beam walk over single-layer graph
- Parameters: `m = 16` (max neighbors), `ef_construction = 200`

---

## 7. Conformal Prediction

> Source file: `rtmdk/memory/conformal.py`

### 7.1 Inductive Conformal Prediction (ICP)

**Calibration set**: $\{(x_i, y_i, s_i)\}$ where $s_i = \text{resonance}(x_i, y_i)$.

**Non-conformity score**:

$$\alpha_i = 1 - s_i \qquad \text{(lower = more conforming)}$$

**Quantile threshold** (Shafer-Vovk exact order statistics):

$$k = \Bigl\lceil (n + 1) \cdot (1 - \alpha_{\text{level}}) \Bigr\rceil$$

$$\text{threshold} = \text{sorted\_scores}[n - k] \quad \text{(k-th largest score)}$$

**Prediction set**:

$$C(x_{n+1}) = \{ y : s(x_{n+1}, y) \geq \text{threshold} \}$$

### 7.2 Coverage Guarantee

**Theorem**. The ICP prediction set satisfies marginal coverage:

$$\mathbb{P}\bigl(\text{target} \in C(x_{n+1})\bigr) \geq 1 - \alpha_{\text{level}}$$

Default $\alpha_{\text{level}} = 0.10$ (90% confidence).

### 7.3 Invalidation After Embedder Training

**Critical constraint** (AGENTS.md): After any embedder training (`train_sot_v2`), the conformal calibrator **must** be reset. Coverage guarantees are void if the embedding distribution shifts without recalibration.

---

## 8. State Space Model Dynamics

> Source: `docs/06_SCIENTIFIC_ARTICLE.md`, `rtmdk/memory/cpen_ode.py`

### 8.1 Backward Euler Discretization

$$\bar{A} = (I - \Delta t \cdot A)^{-1}$$

$$\bar{B} = (I - \Delta t \cdot A)^{-1} \cdot \Delta t \cdot B$$

### 8.2 Evolution Step

$$h_{t+1} = \bar{A} \cdot h_t + \bar{B} \cdot u_t$$

$$y_t = C \cdot h_t + D \cdot u_t$$

### 8.3 Complexity

- SSM dynamics: $O(N \cdot s^2)$ where $s$ = state dimension
- Neural ODE alternative: $O(N^3)$ — SSM is asymptotically superior


## 9. Pipeline & Production Mathematics

> Source files: `rtmdk/pipeline/`, `rtmdk/production/`, `rtmdk/server/`

### 9.1 Streaming Pipeline Executor

6-stage explicit pipeline (`PIPELINE_ARCHITECTURE.md`):

| Stage | Complexity | Description |
|-------|-----------|-------------|
| 1. Embed | $O(D)$ | Compute query embedding |
| 2. Route | $O(1)$ | Cascade router decision |
| 3. Retrieve | $O(\log N \cdot d)$ | Resonance + HNSW search |
| 4. Rerank | $O(\text{top\_k} \cdot s_{\text{dim}})$ | Cross-encoder reranking |
| 5. Calibrate | $O(1)$ | ICP threshold check (precomputed) |
| 6. Explain | $O(\text{top\_k})$ | Result explainability |

**Total latency**:

$$T_{\text{total}} = T_{\text{embed}} + T_{\text{route}} + T_{\text{retrieve}} + T_{\text{rerank}} + T_{\text{calibrate}} + T_{\text{explain}}$$

### 9.2 Cross-Encoder Reranker

`CrossEncoderReranker` (`production/reranker.py`) scores query-passage pairs:

$$\text{score}_i = \text{CrossEncoder}\bigl([\text{query}, \text{passage}_i]\bigr)$$

Results are re-sorted by descending cross-encoder score. Default model: `BAAI/bge-reranker-v2-m3`.

### 9.3 Query Decomposition

`QueryDecomposer` (`production/advanced_retrieval.py` / `memory/rag_quality.py`):

$$Q_{\text{decomposed}} = [q_1, q_2, \ldots, q_m]$$

Each sub-query is retrieved independently and results are merged.

### 9.4 Feedback Loop Scoring

`FeedbackLoop` (`production/feedback_loop.py`):

$$\Delta S = \eta \cdot (\text{quality} - 0.5), \qquad \eta = 0.05$$

$$S_i^{\text{new}} = \text{clip}(S_i + \Delta S, \; 0.0, \; 1.0)$$

**Node quality** (rolling average):

$$\bar{q}_i = \frac{1}{\min(20, |H_i|)} \sum_{r \in H_i[-20:]} r.\text{quality}$$

**Session quality**:

$$\bar{q}_{\text{session}} = \frac{1}{|H_s|} \sum_{r \in H_s} r.\text{quality}$$

### 9.5 Adaptive Decay (Quality-Driven)

From `production/advanced_retrieval.py`:

$$\delta = (0.5 - \text{quality}) \cdot 0.01$$

$$\rho_{\text{new}} = \text{clip}(\rho + \delta, \; \rho_{\min}, \; \rho_{\max})$$

| Quality | Effect on Decay |
|---------|----------------|
| $> 0.5$ | Slower decay (longer memory) |
| $< 0.5$ | Faster decay (shorter memory) |
| $= 0.5$ | No change |

### 9.6 Trust Consensus

$$\text{Weighted\_mean} = \frac{\sum_i \text{rep}_i^2 \cdot \text{embedding}_i}{\sum_i \text{rep}_i^2}$$

**Reputation decay**:

$$\text{rep}_i \leftarrow \text{rep}_i \cdot 0.99 \quad \text{per round}$$

### 9.7 Circuit Breaker

> Source: `rtmdk/support/circuit_breaker.py`

**State machine**:
- **CLOSED**: Normal operation, failures counted
- **OPEN**: Fast-fail after `failure_threshold` consecutive failures
- **HALF_OPEN**: One probe call after `recovery_timeout` seconds

**Transition conditions**:

$$\text{OPEN} \xrightarrow{\text{fail\_count} \geq \text{threshold}} \text{OPEN}$$

$$\text{OPEN} \xrightarrow{t - t_{\text{last\_failure}} \geq \text{recovery\_timeout}} \text{HALF_OPEN}$$

$$\text{HALF_OPEN} \xrightarrow{\text{success}} \text{CLOSED}$$

$$\text{HALF_OPEN} \xrightarrow{\text{failure}} \text{OPEN}$$

Defaults: `failure_threshold = 3`, `recovery_timeout = 30.0s`.

### 9.8 Incremental PCA Projection

> Source: `rtmdk/support/projection.py`

**Manual Oja update** (when sklearn unavailable):

$$\alpha_t = \frac{\eta}{1 + n_{\text{samples}} \cdot \eta \cdot 0.01}$$

$$\mu \leftarrow \mu + \alpha_t \cdot (\bar{x} - \mu)$$

$$\text{latent} = (x - \mu) \cdot P$$

$$\text{reconstructed} = \text{latent} \cdot P^T$$

$$\text{error} = (x - \mu) - \text{reconstructed}$$

$$P \leftarrow P + \alpha_t \cdot \bigl(\text{outer}(x - \mu, \text{latent}) - \text{outer}(\text{error}, \text{latent})\bigr) - \alpha_t \cdot \lambda_{\text{L2}} \cdot P$$

$$P \leftarrow P / \|P\|_{\text{col}} \quad \text{(column normalization)}$$

### 9.9 Rate Limiter (Token Bucket)

> Source: `rtmdk/production/rate_limiter.py`

Token bucket algorithm:

$$\text{tokens} = \min(\text{capacity}, \; \text{tokens} + \text{rate} \cdot \Delta t)$$

Request allowed if $\text{tokens} \geq 1$, then $\text{tokens} \leftarrow \text{tokens} - 1$.

---

## 10. Complexity Analysis

### 10.1 Per-Operation Complexity

| Operation | Complexity | Source File |
|-----------|-----------|-------------|
| Embedding projection | $O(D \cdot d)$ | `support/projection.py` |
| Resonance (1 node) | $O(d)$ | `memory/resonance.py` |
| Search (no index) | $O(N \cdot d)$ | `memory/core.py` |
| Search (with HNSW alias) | $O(N \cdot d)$ | `support/hnsw.py` |
| BM25 search | $O(N \cdot L_{\text{avg}})$ | `support/bm25.py` |
| Cosine similarity | $O(N \cdot D)$ | `memory/core.py` |
| Engram retrieval | $O(E \cdot D)$ | `memory/core.py` |
| Causal traversal | $O(K \cdot d^{\text{hops}})$ | `memory/core.py` |
| Consolidation | $O(N \cdot k \cdot d)$ | `memory/field.py` |
| Decay (all nodes) | $O(N)$ | `memory/scheduler.py` |
| Tension (1 node) | $O(k \cdot d)$ | `memory/topology_manager.py` |
| Self-healing | $O(N \cdot d)$ | `support/healer.py` |
| SSM dynamics | $O(N \cdot s^2)$ | `memory/cpen_ode.py` |
| PC-algorithm | $O(N^2 \cdot |S|)$ | `memory/core.py` |
| HNSW alias build | $O(N^2 \cdot d)$ | `support/hnsw.py` |
| Offline dreaming | $O(N^2)$ | `production/offline_dreamer.py` |
| Trust consensus | $O(P^2)$ | `production/advanced_retrieval.py` |
| Spectral clustering | $O(N^3)$ worst-case | `memory/spectral.py` |
| Kalman filter update | $O(d^2)$ full, $O(d)$ diagonal | `memory/kalman.py` |
| SIF PMI (dense) | $O(n_{\text{valid}}^2)$ | `memory/sot_v2/` |
| SIF PMI (sparse) | $O(n_{\text{nnz}})$ | `memory/sot_v2/` |
| Randomized SVD | $O(n \cdot d^2)$ | `memory/sot_v2/` |
| InfoNCE forward | $O(B \cdot d)$ | `memory/sot_v2/` |
| Conformal calibration | $O(n)$ | `memory/conformal.py` |
| Conformal prediction | $O(1)$ | `memory/conformal.py` |
| Adaptive bandwidth | $O(C \cdot P \cdot N \cdot d)$ | `support/adaptive_bandwidth.py` |
| Cross-encoder rerank | $O(K \cdot s_{\text{dim}}^2)$ | `production/reranker.py` |
| Shard routing | $O(S \cdot d)$ | `memory/routing_manager.py` |
| Shard center update (k-means) | $O(N \cdot S \cdot d \cdot \text{iters})$ | `memory/routing_manager.py` |

### 10.2 RAM Scaling Models

Per-component RAM (`docs/06_SCIENTIFIC_ARTICLE.md`):

| Component | RAM (1K nodes) | RAM (100K nodes) | Scaling |
|-----------|---------------|------------------|---------|
| Nodes ($d=256$) | ~1 MB | ~100 MB | $O(N \cdot d)$ |
| Original embeddings ($D=768$) | ~3 MB | ~300 MB | $O(N \cdot D)$ |
| HNSW alias index | ~2 MB | ~200 MB | $O(N \cdot m)$ |
| BM25 index | ~1 MB | ~100 MB | $O(N \cdot L_{\text{avg}})$ |
| Engrams | ~2 MB | ~200 MB | $O(E \cdot k)$ |
| Causal graph | ~0.5 MB | ~50 MB | $O(N \cdot \bar{\text{deg}})$ |
| Embedding cache | ~3 MB | ~300 MB | $O(N \cdot D)$ |
| **Total** | **~12.5 MB** | **~1250 MB** | **$O(N)$** |

**Shard scaling formula** (`docs/05_FINE_TUNING.md`):

$$\text{num\_shards} = \sqrt{\frac{N}{1000}}$$

Examples:
- 10K nodes → ~3 shards
- 100K nodes → ~10 shards
- 1M nodes → ~32 shards

### 10.3 SIF OOM Boundary

**Dense PMI path** (`np.linalg.svd`): $O(n_{\text{valid}}^2)$ memory.  
**Sparse PMI path** (`scipy.sparse` + `TruncatedSVD`): activates automatically when vocabulary $> 5000$.

---

## 11. Performance Benchmarks

### 11.1 Recall & Latency

| Metric | Value |
|--------|-------|
| Recall@1 (1000 QA) | **95.6%** |
| Recall@5 | ~97% |
| Latency P50 | 83 ms (with LM Studio) / 3–5 ms (no API) |
| Latency P95 | 111 ms |
| RAM (1000 nodes) | ~16 MB |
| Indexing (1000 nodes) | 0.3 s |

### 11.2 Per-Topic Recall

| Topic | Recall |
|-------|--------|
| Biology | 100% |
| Chemistry | 100% |
| History | 100% |
| Literature | 100% |
| Technology | 100% |
| Geography | 99% |
| Health | 96% |
| Science | 96% |
| Physics | 95% |
| Art | 92% |
| **Average** | **97.6%** |

### 11.3 Scalability

| $N$ nodes | Recall@1 | Latency P95 | RAM | Indexing |
|-----------|----------|-------------|-----|----------|
| 200 | 64% | 34 ms | 10 MB | 1.2 s |
| 500 | 60% | 6 ms | 11 MB | 2.8 s |
| 1000 | 95.6% | <1 ms | 16 MB | 0.3 s |
| 10,000 | 100%* | 1.2 ms | 333 MB | 0.8 s |

### 11.4 Ablation Study

| Configuration | R@1 | $\Delta$R@1 |
|-------------|-----|------------|
| Only BM25 | 2.5% | — |
| Only Cosine (FAISS) | 2.5% | — |
| Only Resonance | 92% | +89.5% |
| + Engrams | ~95% | +0–2% |
| + BM25 fallback | 92% | +0% |
| + Cosine hybrid | 94% | +2% |
| + Causal traversal | ~96% | +0–2% |
| **+ All Combined** | **95.6%** | **+3%** |

### 11.5 RAG Comparison

| System | Recall@1 | Recall@5 | Latency P95 | RAM (1K) |
|--------|----------|----------|-------------|----------|
| RTMDK v8.1 | 95.6% | ~97% | 111 ms | 16 MB |
| GraphRAG | 82–90% | 90–95% | 500 ms–2 s | ~200 MB |
| Self-RAG | 80–88% | 88–93% | 300–800 ms | ~100 MB |
| RAFT | 85–90% | 92–96% | 150–400 ms | ~150 MB |
| LightRAG | 78–86% | 85–92% | 100–300 ms | ~80 MB |
| Advanced RAG | 75–85% | 85–92% | 200–500 ms | ~100 MB |
| Naive RAG | 60–75% | 70–85% | 50–200 ms | ~50 MB |

### 11.6 SOT v2.0 Benchmarks

| Method | Recall@5 | MRR | Latency p50 | External deps |
|--------|----------|-----|-------------|---------------|
| Cosine RAG (SBERT) | 1.000 | 0.975 | 0.13 ms | 400 MB model |
| RTMDK + SBERT | 1.000 | 0.964 | 1.74 ms | 400 MB model |
| RTMDK + BGE-M3 | 1.000 | 0.967 | 1.76 ms | 560 MB model |
| RTMDK + SOT v1 | 1.000 | 0.954 | 1.74 ms | None |
| RTMDK + SOT v2 | 1.000 | 0.925 | 1.75 ms | None |
| SOT v2 Hybrid | 0.985 | 0.934 | 0.50 ms | None |
| SOT v2 Quantum | 0.985 | 0.912 | 2.03 ms | None |

### 11.7 Hard Dataset (comprehensive_500)

| Method | Recall@1 | Recall@5 | MRR |
|--------|----------|----------|-----|
| SBERT (teacher) | 0.181 | 1.000 | 0.591 |
| SOT v2 | 0.081 | 0.935 | 0.374 |

### 11.8 Pipeline Stress Test

| Metric | 1K nodes | 5K nodes | 10K nodes |
|--------|----------|----------|-----------|
| Insert throughput | ~10K/s | 7,877/s | 7,085/s |
| Query P50 | <1 ms | 0.96 ms | 1.21 ms |
| Query P95 | <1 ms | 1.36 ms | 1.65 ms |
| Query P99 | <1 ms | 8.14 ms | 1.89 ms |
| RAM | ~16 MB | ~299 MB | ~333 MB |

### 11.9 SOT Enhancement Track

| Method | Recall@1 (1000 QA) |
|--------|-------------------|
| BM25 | 0.691 |
| Byte v1 | 0.224 |
| Word v1 | 0.356 |
| Word + SBERT bootstrap | **0.799** |
| Word + FastText bootstrap | **0.769** |


## 12. Theorems & Proofs

### Theorem 1 — RTMDK Dominates Naive RAG

**Statement**. For any query $q$ and document corpus $D$:

$$\text{Recall@1}(\text{RTMDK}) \geq \text{Recall@1}(\text{RAG})$$

**Proof sketch**.

$$
\begin{aligned}
\text{RAG}(q) &= \arg\max_{d \in D} \; \text{sim}(q, d) \\[4pt]
\text{RTMDK}(q) &= \arg\max_{d \in D} \; \bigl[K_{\text{spatial}}(q,d) \cdot K_{\text{phase}}(q,d) \cdot A(d) \cdot S(d)\bigr]
\end{aligned}
$$

1. $K_{\text{spatial}} \geq \text{sim}$ because Gaussian is a monotonic transform of cosine.
2. $K_{\text{phase}} \in [0, 1]$ provides additional non-negative signal.
3. $A \cdot S \in [0, 1]$ are dynamic adaptive weights.
4. Engrams provide pattern completion beyond single-document matching.
5. Causal traversal adds causally-linked nodes to the candidate set.

**Corollary**. Equality holds only when $K_{\text{phase}} \equiv 1$ and $A \equiv S \equiv 1$ (i.e., RTMDK reduces to RAG).

*Source: `docs/06_SCIENTIFIC_ARTICLE.md`*

---

### Theorem 2 — Decay Convergence

**Statement**. For decay rate $\rho \in (0, 1)$:

$$S_i(t) = S_i(0) \cdot \rho^t \xrightarrow[t \to \infty]{} 0$$

**Proof**. Since $|\rho| < 1$, the geometric sequence $\rho^t$ converges to 0. Salience $S_i(t)$ is bounded above by $S_i(0) \leq 1$ and monotonically decreasing. By the monotone convergence theorem, $S_i(t) \to 0$.

---

### Theorem 3 — Half-Life Formula

**Statement**. The half-life $t_{1/2}$ of exponential decay with rate $\rho$ is:

$$t_{1/2} = \frac{\ln(0.5)}{\ln(\rho)}$$

**Proof**. Solve $S_i(0) \cdot \rho^{t_{1/2}} = S_i(0) / 2$:

$$\rho^{t_{1/2}} = \frac{1}{2} \;\Rightarrow\; t_{1/2} \cdot \ln(\rho) = \ln(0.5) \;\Rightarrow\; t_{1/2} = \frac{\ln(0.5)}{\ln(\rho)}$$

Since $\rho \in (0, 1)$, $\ln(\rho) < 0$ and $t_{1/2} > 0$.

---

### Theorem 4 — Spectral Embedding Minimality (von Luxburg 2007)

**Statement**. The $k$-dimensional embedding minimizing the Dirichlet energy:

$$E(X) = \frac{1}{2} \sum_{i,j} w_{ij} \|x_i - x_j\|^2$$

subject to $X^T X = I$, is given by the $k$ smallest eigenvectors of $L_{\text{sym}}$.

**Proof sketch**. The Dirichlet energy can be rewritten as:

$$E(X) = \text{tr}(X^T L X)$$

where $L = D - W$ is the unnormalized Laplacian. By the Rayleigh-Ritz theorem, the minimizers of $\text{tr}(X^T L X)$ subject to orthonormality constraints are the eigenvectors corresponding to the smallest eigenvalues. For the normalized Laplacian $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$, the same argument applies after the change of variables $Y = D^{1/2} X$.

*Source: `docs/SOT_V2_THEORY.md`*

---

### Theorem 5 — Orthogonal Procrustes (Schönemann 1966)

**Statement**. Given $X, Y \in \mathbb{R}^{n \times d}$, the orthogonal matrix $R$ minimizing $\|XR - Y\|_F^2$ is:

$$R = U \, V^T \quad \text{where} \quad U \, \Sigma \, V^T = \text{SVD}(X^T Y)$$

**Proof sketch**. Expand the objective:

$$\|XR - Y\|_F^2 = \text{tr}(R^T X^T X R) - 2 \, \text{tr}(R^T X^T Y) + \text{tr}(Y^T Y)$$

Since $R$ is orthogonal, the first term is constant. Maximizing $\text{tr}(R^T X^T Y)$ subject to $R^T R = I$ is solved by setting $R = U V^T$ where $X^T Y = U \Sigma V^T$ is the SVD. This follows from the von Neumann trace inequality.

*Source: `docs/SOT_V2_THEORY.md`, `rtmdk/memory/sot_v2/`*

---

### Theorem 6 — Hybrid Fusion Optimality (Clinchant & Gaussier 2010)

**Statement**. If retrieval errors from dense and sparse systems are conditionally independent with equal variance, the optimal fusion weight is $\alpha = 0.5$.

**Proof sketch**. Let $\varepsilon_d$ and $\varepsilon_s$ be independent errors with $\text{Var}(\varepsilon_d) = \text{Var}(\varepsilon_s) = \sigma^2$. The fused score is:

$$S_{\text{hybrid}} = \alpha \cdot S_d + (1 - \alpha) \cdot S_s$$

The variance of the fused error is:

$$\text{Var}(\varepsilon_{\text{hybrid}}) = \alpha^2 \sigma^2 + (1 - \alpha)^2 \sigma^2 = \sigma^2 \bigl(\alpha^2 + (1 - \alpha)^2\bigr)$$

Minimizing with respect to $\alpha$:

$$\frac{d}{d\alpha} \bigl(\alpha^2 + (1 - \alpha)^2\bigr) = 2\alpha - 2(1 - \alpha) = 0 \;\Rightarrow\; \alpha = 0.5$$

*Source: `docs/SOT_V2_THEORY.md`*

---

### Theorem 7 — MI-Subword Optimality (Gage 1994; Sennrich et al. 2016)

**Statement**. Greedy merge of subword pairs maximizing mutual information yields a tokenization that minimizes the cross-entropy of the corpus under a unigram language model.

**Proof sketch**. At each merge step, the change in corpus cross-entropy is:

$$\Delta L = -\bigl[ |C|_{ab} \log P(c) - |C|_a \log P(a) - |C|_b \log P(b) \bigr]$$

where $c = a \oplus b$ is the merged token. By definition of mutual information:

$$\text{MI}(a, b) = \log \frac{P(a, b)}{P(a) P(b)}$$

Maximizing MI is equivalent to maximizing $P(a, b) / (P(a) P(b))$, which minimizes $\Delta L$ (the most negative $\Delta L$ gives the largest entropy reduction). Greedy application preserves this property at each step.

*Source: `docs/SOT_V2_THEORY.md`*

---

### Theorem 8 — SIF MLE (Arora et al. 2017)

**Statement**. Under the random-walk discourse model where the context vector $c_t$ executes a random walk in embedding space, the MLE for a sentence embedding is the Smooth Inverse Frequency weighted average of word embeddings.

**Proof sketch**. The discourse model assumes:

$$P(w \mid c_t) \propto \exp(\langle v_w, c_t \rangle)$$

Given a sentence $s = (w_1, \ldots, w_n)$, the log-likelihood is:

$$\log P(s \mid c) = \sum_{i=1}^{n} \log P(w_i \mid c) = \sum_{i=1}^{n} \bigl(\langle v_{w_i}, c \rangle - \log Z(c)\bigr)$$

Taking the derivative with respect to $c$ and setting to zero:

$$\nabla_c \log P(s \mid c) = \sum_{i=1}^{n} v_{w_i} - n \cdot \nabla_c \log Z(c) = 0$$

The partition function gradient $\nabla_c \log Z(c)$ can be approximated by the corpus average $\sum_w p(w) v_w$. The SIF weight $a / (a + p(w))$ emerges from a more careful analysis of the random-walk stationary distribution, correcting for word frequency bias.

*Source: `docs/SOT_V2_THEORY.md`, Arora et al. (2017) "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"*

---

### Theorem 9 — ICP Marginal Coverage (Shafer & Vovk 2008)

**Statement**. For any exchangeable calibration set and significance level $\alpha$, the prediction set produced by Inductive Conformal Prediction satisfies:

$$\mathbb{P}\bigl(Y_{n+1} \in C(X_{n+1})\bigr) \geq 1 - \alpha$$

**Proof sketch**. Let $\alpha_i = 1 - s_i$ be the non-conformity scores of the calibration set and $\alpha_{n+1} = 1 - s_{n+1}$. Under exchangeability, the rank of $\alpha_{n+1}$ among all $n+1$ scores is uniformly distributed. The prediction set includes $y$ if and only if $\alpha_{n+1}$ is among the $\lceil (n+1)(1-\alpha) \rceil$ smallest non-conformity scores. Since the rank is uniform:

$$\mathbb{P}\bigl(\alpha_{n+1} > \hat{q}\bigr) \leq \alpha$$

where $\hat{q}$ is the empirical $(1-\alpha)$-quantile. Therefore:

$$\mathbb{P}\bigl(Y_{n+1} \in C(X_{n+1})\bigr) = 1 - \mathbb{P}\bigl(\alpha_{n+1} > \hat{q}\bigr) \geq 1 - \alpha$$

*Source: `rtmdk/memory/conformal.py`, `docs/MATH_BACKLOG.md`*

---

## Appendix A: Configuration Presets with Mathematical Parameters

| Preset | `latent_dim` | `decay_rate` | Expected RAM | Expected Latency | Max Nodes |
|--------|-------------|-------------|-------------|-----------------|-----------|
| `local()` | 256 | 0.999 | ~16 MB | ~5 ms | 10K |
| `production()` | 256 | — | ~50 MB | ~6 ms | 100K |
| `research()` | 512 | — | ~200 MB | ~50 ms | $\infty$ |
| `enterprise()` | — | — | ~250 MB/shard | ~15 ms | 500K+ |
| `streaming()` | — | — | ~30 MB | ~3 ms | 50K |

---

## Appendix B: Key Parameter Quick Reference Table

| Parameter | Symbol | Default | Range | File |
|-----------|--------|---------|-------|------|
| bandwidth | $\beta$ | 1.0 | [0.1, 10.0] | `config.py:445` |
| phase_coupling | $\kappa$ | 0.3 | [0.0, 1.0] | `config.py:444` |
| decay_rate | $\rho$ | 0.997 | (0, 1) | `config.py:462` |
| tension_threshold | $\tau^*$ | 0.15 | [0.05, 0.5] | `config.py:464` |
| min_response | — | 0.005 | [0, 1] | `config.py:469` |
| gate_temperature | $\gamma$ | 0.15 | (0, $\infty$) | `config.py:498` |
| hybrid_alpha | $\alpha$ | 1.0 | [0, 1] | `config.py:489` |
| conformal_alpha | $\alpha_{\text{level}}$ | 0.10 | (0, 1) | `config.py:447` |
| ball_radius | $R$ | 0.85 | (0, $\infty$) | `config.py:642` |
| bm25_k1 | $k_1$ | 1.5 | (0, $\infty$) | `config.py:484` |
| bm25_b | $b$ | 0.75 | [0, 1] | `config.py:485` |
| learning_rate | $\eta$ | 0.05 | (0, $\infty$) | `feedback_loop.py` |
| process_noise | $Q$ | 0.01 | (0, $\infty$) | `kalman.py` |
| measurement_noise | $R$ | 0.1 | (0, $\infty$) | `kalman.py` |
| sif_a | $a$ | 0.001 | (0, $\infty$) | `sot_v2/` |
| failure_threshold | — | 3 | $\mathbb{N}$ | `circuit_breaker.py` |
| recovery_timeout | — | 30.0 s | (0, $\infty$) | `circuit_breaker.py` |

---

## Appendix C: Optimization Summary Table

| Optimization | Location | Purpose | Trigger Condition |
|-------------|----------|---------|-------------------|
| Sparse PMI path | `sif_embedder.py` | Avoid $O(n^2)$ dense matrix | vocab > 5000 |
| Dense PMI path | `sif_embedder.py` | Faster SVD for small vocab | vocab $\leq$ 5000 |
| Randomized SVD | `sif_embedder.py` | $O(nd^2)$ vs $O(n^3)$ | $n > 2000$ |
| Power iteration for first PC | `sif_embedder.py` | Fast dominant eigenvector | Always (default) |
| InfoNCE logit shifting | `sif_embedder.py` | Numerical stability in softmax | Always |
| Low-rank quantum query | `quantum.py` | Fast matrix-vector product | Quantum mode enabled |
| Min-max score normalization | `hybrid_retriever.py` | Align dense/sparse scales | Hybrid mode enabled |
| Orthonormalization of $W$ | `sif_embedder.py` | Prevent embedding collapse | Contrastive training |
| Int8 quantization | `resonance.py` | Accelerated resonance | `use_int8 = True` |
| Diagonal Kalman approximation | `kalman.py` | $O(d)$ vs $O(d^2)$ | `diagonal_approx = True` |
| Adaptive bandwidth | `adaptive_bandwidth.py` | Auto-tuned kernel width | Every 200 queries |
| Sparse shard routing | `routing_manager.py` | Sub-linear retrieval | `sparse_routing = True` |
| Circuit breaker fast-fail | `circuit_breaker.py` | Prevent cascade failures | 3+ consecutive failures |
| Incremental PCA | `projection.py` | Online projection update | Every 50 samples |
| Tier-specific decay | `scheduler.py` | Context-aware forgetting | Always (by node tier) |
| Tension cache | `field.py` | Avoid redundant computation | Step % 50 hit-rate check |
| Meta-kernel adaptation | `field.py` | Auto-tune $\beta, \kappa$ | Every 5 steps |
| Predictive coding | `scheduler.py` | Trigger consolidation on surprise | Free energy > 0.3 |
| Low-rank field compression | `field.py` | Reduce memory footprint | `compression_freq` steps |
| Self-healing | `healer.py` | Repair orphaned/dead nodes | `healing_check_freq` steps |

---

*Document version: RTMDK v8.3.0*  
*Last updated: 2026-05-22*  
*Math extracted from: `rtmdk/`, `docs/`, `tests/`, `scripts/`, `README.md`, `AGENTS.md`*
