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
from rtmdk.memory.config import RTMDKConfig
from rtmdk.memory.core import RTMDKMemory

cfg = RTMDKConfig(
    latent_dim=384,
    sot_v2_enabled=True,
    sot_v2_a=0.01,           # SIF smoothing parameter
    sot_v2_window=5,         # Co-occurrence window
    sot_v2_remove_pc=True,   # Remove first principal component
    sot_v2_hybrid_alpha=0.5, # BM25+SIF fusion weight
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
| `sot_v2_a` | `0.01` | SIF smoothing (`a / (a + p(w))`). Lower = more weight to rare words. |
| `sot_v2_window` | `5` | Sliding window for co-occurrence counts. Larger = more semantic neighbours. |
| `sot_v2_remove_pc` | `True` | Remove first principal component from sentence embeddings. Usually helps. |
| `sot_v2_hybrid_alpha` | `0.5` | Dense/Sparse fusion. `1.0` = SIF only, `0.0` = BM25 only. |

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

**Rule of thumb:** start with `a=0.01`, `window=5`, `alpha=0.5`.  Increase `window` if your documents are long; decrease `a` if your vocabulary is very large.

## Known Limitations

- **Morphologically rich languages** (Turkish, Finnish) may need subword tokenisation (`MI_SubwordTokenizer`) instead of word-level.
- **CJK scripts** work but word-level splitting is less meaningful than for Latin scripts.
- **No pre-trained knowledge** — SOT v2.0 only knows what is in your corpus.  It will not understand out-of-vocabulary domain terms unless they appear in the training data.

## Further Reading

- [SOT v2.0 Theory](SOT_V2_THEORY.md) — mathematical derivations and proofs
- [API Reference](01_API_REFERENCE.md)
- [Benchmark Script](../scripts/bench_sot_v2.py)
