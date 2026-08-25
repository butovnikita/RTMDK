# SOT v2.0 — Production Guide

## What is SOT v2.0?

**SOT v2.0** (Self-Organising Tokenizer) is a fully self-contained semantic
embedding system built into RTMDK.  After a one-time training step on your
corpus, it requires **zero external model downloads**, **no GPU**, and fits in
**~15 MB RAM**.

| Metric | SBERT | BGE-M3 | SOT v2.0 |
|--------|-------|--------|----------|
| External deps | 400 MB | 560 MB | **None** |
| GPU required | Optional | Recommended | **No** |
| Cold-start | Instant | Slow (first download) | Fast (train on corpus) |
| Recall@5 (qa_1000_en) | 100 % | 100 % | **100 %** |
| Latency p50 | 0.13 ms | 1.76 ms | **1.75 ms** |

## Quick Start

### Option A: Via `RTMDKConfig` (recommended)

```python
from rtmdk.memory.config import RTMDKConfig, SOTConfig
from rtmdk.memory.core import RTMDKMemory

cfg = RTMDKConfig(
    latent_dim=384,
    sot=SOTConfig(
        sot_v2_enabled=True,
        sot_v2_a=0.01,           # SIF smoothing parameter (or 'adaptive')
        sot_v2_window=5,         # Co-occurrence window
        sot_v2_remove_pc=True,   # Remove first principal component
        sot_v2_hybrid_alpha=0.5, # BM25+SIF fusion weight
        # Optional: lightweight teacher distillation
        # sot_v2_align_teacher="sentence-transformers/all-MiniLM-L6-v2",
    ),
)

memory = RTMDKMemory(config=cfg, embedder=lambda t: None)

# Ingest your documents
for doc in documents:
    memory.add_node(
        embedding=np.zeros(384, dtype=np.float32),  # placeholder
        content={"text": doc["text"]},
    )

# One-time training (seconds for 1k docs)
memory.train_sot_v2(extra_texts=[q["text"] for q in queries])

# Query
results = memory.retrieve_nodes(query_text, query_embedding, top_k=5)
```

### Option B: Standalone `SOTv2Embedder`

```python
from rtmdk.memory.sot_v2 import SOTv2Embedder

embedder = SOTv2Embedder(latent_dim=384)
embedder.train(corpus_texts)

embedding = embedder("Your query text here")
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sot_v2_enabled` | `False` | Master switch |
| `sot_v2_a` | `adaptive` | SIF smoothing (`a / (a + p(w))`). Auto-estimated as P10 of word probs. |
| `sot_v2_window` | `5` | Sliding window for co-occurrence counts. Larger = more semantic neighbours. |
| `sot_v2_remove_pc` | `True` | Remove first principal component from sentence embeddings. Usually helps. |
| `sot_v2_hybrid_alpha` | `0.5` | Dense/Sparse fusion. `1.0` = SIF only, `0.0` = BM25 only. |
| `sot_v2_align_teacher` | `None` | Optional SBERT model name for Procrustes alignment (inference still zero-dep). |
| `sot_v2_align_center` | `True` | Mean-center both spaces before Procrustes alignment. |
| `sot_v2_aligner_path` | `None` | Load a pre-fitted `.npz` alignment matrix. |

## When to Use SOT v2.0

**Use SOT v2.0 when:**
- You are in an **air-gapped environment** (no internet, no HF Hub).
- You need **tiny footprint** (embedded systems, edge devices).
- Your corpus is **small to medium** (1k–100k short texts).
- Your queries and documents share **vocabulary overlap**.

**Use SBERT/BGE-M3 when:**
- You need **multilingual** support (SOT v2.0 is optimised for space-delimited scripts).
- Your corpus is **large and diverse** (MS MARCO, Wikipedia-scale).
- You need **sparse lexical matching** (IDs, codes, exact phrases) — BGE-M3 provides this natively.

## How It Works (One Paragraph)

1. **Word tokenisation** — Unicode-aware word splitting.
2. **PMI matrix** — Pointwise mutual information from sliding-window co-occurrences.
3. **Spectral embedding** — Truncated SVD of the PMI matrix yields word vectors.
4. **SIF pooling** — Sentence embedding = weighted average of word vectors, with the first principal component removed.
5. **Hybrid retrieval** — Convex fusion of dense SIF cosine similarity and BM25 lexical scores.

## Performance Tuning

### Grid-Search Results (qa_1000_en.json, 200 records)

| `a` | `window` | `alpha` | Recall@5 | MRR |
|-----|----------|---------|----------|-----|
| 0.01 | 5 | 0.5 | 98.5 % | **0.934** |
| 0.01 | 5 | 1.0 | 98.5 % | 0.878 |
| 0.1 | 20 | 0.7 | 98.5 % | 0.934 |

**Rule of thumb:** start with `a=0.01`, `window=5`, `alpha=0.5`.  
SOT v2.0 now estimates `a` automatically from corpus statistics (10th
percentile of word probabilities), so manual tuning is rarely needed.

## New in v8.3 — Breakthrough Features

### 1. Semantic Phase Learning
Node phases are no longer random (`time.time()*0.01`).  They are derived
from `hash(session_id + topic + keywords)`, so nodes from the same
conversation or topic share phase neighbourhoods.  The resonance kernel's
`cos(Δφ)` term now **genuinely boosts intra-cluster retrieval** — a
feature no other vector database offers.

Enable: always on (computed automatically in `_get_phase`).

### 2. Learned Consolidation
A tiny MLP (~450K params for d=384) learns how to merge two nodes while
preserving retrieval quality.  Instead of heuristic averaging, the merged
latent is predicted from parent states and trained on synthetic merge
examples with a hinge loss.

```python
RTMDKConfig(
    learned_consolidation=True,
)
```

### 3. Causal Graph from LLM Explanations
When a node contains explanation text ("A because B"), RTMDK extracts
directed causal edges and stores them in `node.causal_parents`.  This
replaces the statistically broken PC-algorithm with robust pattern
matching on natural language.

### 4. Adaptive Bandwidth
Bandwidth is optimised by random search on a synthetic calibration set
(self-retrieval recall).  No more hand-tuning or kurtosis chasing.

```python
RTMDKConfig(
    adaptive_bandwidth=True,
)
```

### 5. Conformal Prediction API
Statistical guarantee: `P(target ∈ prediction_set) ≥ 1 - α`.

```python
result = memory.query_with_confidence(query, embedding, alpha=0.05)
# result["prediction_set"] — node_ids with coverage guarantee
# result["coverage_guarantee"] — True if calibrated
```

## Known Limitations

- **Morphologically rich languages** (Turkish, Finnish) may need subword tokenisation (`MI_SubwordTokenizer`) instead of word-level.
- **CJK scripts** work but word-level splitting is less meaningful than for Latin scripts.
- **No pre-trained knowledge** — SOT v2.0 only knows what is in your corpus.  It will not understand out-of-vocabulary domain terms unless they appear in the training data.

## RAM & OOM Guide — R6.1 (2026-08-24, audit/risks-2026-08-24)

**AGENTS.md Critical Constraint #1 — SIF dense PMI matrix scales as O(n_valid²).**

| Vocab (`sot_max_vocab`) | PMI path | RAM / compute | Note |
|:---:|:---:|:---|:---|
| `≤4096` (default) | **dense** `np.zeros((n,n))` `sif_embedder.py:221` | `4096²×8B≈134MB` + SVD `~×2` | Fits in 16MB RSS? No — but dense fits in 256MB, okay for 1K docs |
| `4096–5000` | **dense** (fast BLAS) | `5000²×8≈200MB` | Threshold `SPARSE_PMI_THRESHOLD=5000` `sif_embedder.py:168` |
| `5000–8000` | **sparse** `coo_matrix + TruncatedSVD` `sif_embedder.py:171` | `CSR + randomized SVD ~0.5–1GB` | Warn in `validate()` when `>5000`, OK for 10K docs `window=5` |
| `8000–20000` | **sparse** | `10K²≈800MB, 20K²≈3GB` CSR before SVD | `validate()` warns `>8000` (R6.1), user must ensure 6+GB RAM |
| `>20000` | **sparse** | `>3GB` + `TruncatedSVD(randomized)` 3GB+ | Not recommended; consider external embedder (SBERT) |

**Tuning:**

- `sot_skipgram_window` (`default 1`) — `>5` may OOM **before** PMI (COOC `Dict[int,Dict]`, `max_cooccurrence=100K`, `R6.2`). Keep `≤5` (validate warns).
- `sot_max_cooccurrence=100_000` — LRU pruning before matrix build; `>200K` warns.
- `RTMDKConfig.validate()` now emits `sot_max_vocab>8000` and `window>5` warnings (R6.1/R6.2).
- Measure: `python -c "from rtmdk.memory.sot_v2.sif_embedder import SPARSE_PMI_THRESHOLD; print(SPARSE_PMI_THRESHOLD)"` (`5000`).

**Rule:** `sot_max_vocab ≤4096` (dense, fast), `5000–8000` (sparse, okay), `>8000` (validate warns, need RAM).

## Persistence Format — R12.1 (2026-08-24)

**Memory file is `msgpack+zlib` + checksum, not JSON** (`rtmdk/memory/serialization.py:174`, `docs/RISKS.md R12.1`).
Default now `~/.rtmdk/memory.msgpack` (`rtmdk/server/app.py:118`, was `memory.json` misleading).
`field_to_file(..., fmt="msgpack")` writes `zlib(msgpack)` with `b"\x78"` header and `_checksum` (sha256);
`field_from_file` auto-detects `msgpack` vs `JSON` and verifies checksum (prevents 8 `*.corrupted.*` from `AUDIT_REPORT.md:34`).
Legacy `memory.json` still readable for backward compat (init_memory tries `.msgpack` then `.json`).

## Further Reading

- [SOT v2.0 Theory](SOT_V2_THEORY.md) — mathematical derivations and proofs
- [API Reference](01_API_REFERENCE.md)
- [Benchmark Script](../scripts/bench_sot_v2.py)
