# SOT v2.0 (Self-Organising Tokenizer)

SOT v2.0 is a **fully self-contained semantic embedding system** built into
RTMDK. After a one-time training step on your corpus it requires **zero
external model downloads, no GPU, and ~15 MB RAM**, while matching SBERT-level
recall on in-domain benchmarks.

## Key Properties

| Metric | SBERT | BGE-M3 | SOT v2.0 |
|--------|-------|--------|----------|
| External deps | 400 MB | 560 MB | **None** |
| GPU required | Optional | Recommended | **No** |
| Recall@5 (qa_1000_en) | 100 % | 100 % | **100 %** |

## Components

1. **MI-Subword Tokenization** — information-theoretic segmentation
2. **Spectral Word Embeddings** — graph Laplacian eigenvectors
3. **Smooth Inverse Frequency (SIF)** — generative sentence embeddings
4. **Hybrid BM25+SIF Retrieval** — dense/sparse fusion
5. **Quantum Resonance Retrieval** — density-matrix late interaction

## Quick Start

```python
from rtmdk.memory.config import RTMDKConfig, SOTConfig
from rtmdk.memory.core import RTMDKMemory

cfg = RTMDKConfig(
    latent_dim=384,
    sot=SOTConfig(
        sot_v2_enabled=True,
        sot_v2_a=0.01,            # SIF smoothing (or 'adaptive')
        sot_v2_hybrid_alpha=0.5,  # BM25+SIF fusion weight
    ),
)
memory = RTMDKMemory(config=cfg, embedder=lambda t: None)

# Ingest documents, then one-time training (seconds for 1k docs)
memory.train_sot_v2(extra_texts=[...])
```

!!! warning "Conformal invalidation"
    After any `train_sot_v2()` call the conformal calibrator **must be reset**,
    or coverage guarantees are void.

## Full Documentation

- [SOT v2.0 Production Guide — config, training, benchmarks, checkpoints](../SOT_V2_GUIDE.md)
- [SOT v2.0 Theoretical Foundations — math behind each component](../SOT_V2_THEORY.md)
