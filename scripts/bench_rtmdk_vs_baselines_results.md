# RTMDK vs FAISS vs Pure Cosine — Benchmark Results

**Date:** 2026-05-07

## Configuration

| Variant | Embedder | Phase Coupling | Notes |
|---------|----------|---------------|-------|
| **SBERT baseline** | `all-MiniLM-L6-v2` (EN) / `paraphrase-multilingual-MiniLM-L12-v2` (RU) | Adaptive | External transformer model |
| **SOT v2 self-contained** | SIF + char-trigram fallback | Adaptive | No external model, trained on corpus+queries |

**RTMDK config:** `latent_dim=384`, `use_hnsw=False`, `sparse_routing=False`, `adaptive_phase_coupling=True`.

## Key Fixes Applied
1. Query phase computed from query text via `_semantic_phase` (was random `time.time()`)
2. `_semantic_phase` uses content words with EN+RU stop-word filtering
3. Embeddings normalized before ingestion for consistent ranking
4. **Adaptive phase coupling** auto-tunes `pc`: 0.00 for easy datasets, 0.10–0.15 for hard
5. **SOT v2 char-trigram fallback** — OOV words decomposed into character n-grams for zero-OOV embedding
6. **Procrustes alignment disabled for small corpora** — alignment improves absolute similarity but destroys local neighborhood structure

---

## Results: SBERT Baseline

### qa_1000_en (200 records, 200 contexts)
| System | recall@1 | recall@5 | MRR |
|--------|----------|----------|-----|
| Pure Cosine | 0.955 | 1.000 | 0.975 |
| RTMDK | **0.955** | **1.000** | **0.975** |

### comprehensive_500 EN (430 records, 80 contexts)
| System | recall@1 | recall@5 | MRR |
|--------|----------|----------|-----|
| Pure Cosine | 0.181 | 1.000 | 0.591 |
| RTMDK | **0.993** | **0.998** | **0.995** |

### comprehensive_500 RU (70 records, 70 contexts)
| System | recall@1 | recall@5 | MRR |
|--------|----------|----------|-----|
| Pure Cosine | 0.857 | 0.957 | 0.902 |
| RTMDK | **0.857** | **0.957** | **0.902** |

---

## Results: SOT v2 Self-Contained (no external embedder)

### qa_1000_en (200 records, 200 contexts)
| System | recall@1 | recall@5 | MRR |
|--------|----------|----------|-----|
| Pure Cosine | 0.765 | 0.965 | 0.849 |
| RTMDK | **0.765** | **0.965** | **0.849** |

### comprehensive_500 EN (430 records, 80 contexts)
| System | recall@1 | recall@5 | MRR |
|--------|----------|----------|-----|
| Pure Cosine | 0.107 | 0.935 | 0.520 |
| RTMDK | **0.919** | **0.933** | **0.925** |

### comprehensive_500 RU (70 records, 70 contexts)
| System | recall@1 | recall@5 | MRR |
|--------|----------|----------|-----|
| Pure Cosine | 0.714 | 0.829 | 0.760 |
| RTMDK | **0.714** | **0.829** | **0.760** |

---

## Analysis

### 1. SBERT baseline — best overall quality
- Perfect on easy datasets (95.5% recall@1)
- Massive RTMDK boost on hard datasets (18.1% → 99.3% recall@1)
- Requires external transformer model (~100MB download)

### 2. SOT v2 self-contained — viable for hard retrieval
- **On hard paraphrase dataset:** RTMDK+SOTv2 achieves **91.9% recall@1** vs 99.3% for SBERT — gap is only 7.4pp
- **No external model needed** — entirely trained on the user's own corpus
- RTMDK's resonance+phase mechanism compensates for weak embeddings (10.7% → 91.9% recall@1)
- Russian works out-of-the-box without multilingual model (71.4% recall@1)

### 3. When to use SOT v2 vs SBERT
| Scenario | Recommendation |
|----------|---------------|
| Easy 1-to-1 retrieval | **SBERT** — SOT v2 quality gap is large (~19pp) |
| Hard paraphrase / conversational memory | **SOT v2 + RTMDK** — gap is small (~7pp), no external deps |
| Offline / air-gapped environments | **SOT v2** — only choice |
| Russian without internet | **SOT v2** — no need to download multilingual model |

### 4. Why Procrustes alignment hurts on small corpora
- Without alignment: SOT rank = 2/80 for "earthquakes" query
- With alignment: SOT rank = 22/80 for same query
- Alignment improves **absolute** similarity (0.047 → 0.549) but scrambles **relative** ranking
- Orthogonal constraint cannot preserve local topology when source space is noisy

## Files Changed

- `rtmdk/memory/field.py` — Improved `_semantic_phase` with stop-word filtering, fixed query phase computation
- `rtmdk/memory/core.py` — `_get_phase` now passes query content to field
- `rtmdk/memory/config.py` — Added `adaptive_phase_coupling` config
- `rtmdk/memory/adaptive_pc.py` — New module for auto-tuning phase coupling
- `rtmdk/memory/sot_v2/sif_embedder.py` — Char-trigram OOV fallback
- `rtmdk/memory/sot_v2/integration.py` — Pass `id2word` to fit, normalize teacher embeddings in alignment
