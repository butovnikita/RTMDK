# RTMDK Development Backlog

> Last updated: 2026-05-01
> Current version: 8.2.0
> Test status: 273 passed, 1 skipped

---

## Executive Summary

RTMDK v8.2 is production-ready for single-node deployment (10K–100K nodes, sub-10ms latency).
The backlog below targets **enterprise scale** (1M+ nodes), **cost reduction**, and **deployment ergonomics**.

---

## Track 1: Quantization (fp16 Done, int8 Backlog)

**Goal:** Reduce memory footprint 2–4× with <1% recall degradation.

**Current state (fp16 shipped):**
- `quantization: "none" | "fp16"` implemented via `QuantizationHelper`
- `node.latent_pos` and `_cached_positions` stored as `float16` when enabled
- 10K nodes × 256d = ~9.8 MB RAM (2× reduction)
- R@1 = 100.0%, R@5 = 99.88% vs float32 baseline (negligible degradation)

**Future: int8 quantization:**
- `int8_global` (scale=1/127) gives 91.6% R@1 — acceptable for some workloads
- `int8_per_dim` gives 98.0% R@1 — better, but requires per-dimension scale tracking
- Blocker: `node.latent_pos` as `int8` breaks many arithmetic paths in consolidation / ODE evolve. Needs `MemoryNode` property wrapper or explicit dequantize at every math site.

**Acceptance criteria:**
- [x] Implement `QuantizationHelper` with fp16 mode
- [x] Benchmark R@1 on 10K nodes: ≥ 99.5% vs float32 baseline
- [x] Add config flag `quantization: "none" | "fp16"`
- [ ] int8 mode (deferred to v8.3 — requires broader `MemoryNode` refactor)

**Effort:** Low (2–3 days) — fp16 done in 1 session
**Impact:** Very High — unlocks mobile/edge deployment

---

## Track 2: Tiered Storage (Hot / Warm / Cold)

**Goal:** Support unlimited node count without proportional RAM growth.

**Current state:**
- All nodes live in RAM (`self.nodes` dict + numpy caches)
- 10K nodes = 30 MB; 1M nodes = ~3 GB — impractical on consumer hardware

**Target state:**
- **Hot** (top 1% by query frequency) → RAM
- **Warm** (next 9%) → memory-mapped disk (`numpy.memmap`)
- **Cold** (remaining 90%) → compressed msgpack snapshots
- Query path checks hot first, then warm, then cold (lazy load)

**Acceptance criteria:**
- [ ] Implement `TieredNodeStore` with hot/warm/cold tiers
- [ ] Benchmark: 1M nodes on 16 GB RAM with p99 query < 20 ms
- [ ] Auto-promotion/demotion based on access frequency (LFU cache)
- [ ] Graceful degradation: if cold node requested, load + promote to warm

**Effort:** High (1–2 weeks)
**Impact:** Very High — enterprise sales enabler

---

## Track 3: Query Cache + Dynamic top_k

**Goal:** Further reduce LLM token spend and latency for repetitive traffic.

**Current state:**
- Every query runs full resonance computation
- Fixed `top_k=5` regardless of result confidence

**Target state:**
- **Query Cache:** LRU cache keyed by embedding hash → cached top-k result
  - Hit rate expected 60–80% for chatbots with repeated questions
  - Latency: cache hit < 0.1 ms vs 1.3 ms compute
- **Dynamic top_k:**
  - If top-1 score > 0.95 → return 1 node (saves 4× context tokens)
  - If top-1 score 0.80–0.95 → return 3 nodes
  - If top-1 score < 0.80 → return 5 nodes (current default)

**Acceptance criteria:**
- [ ] Implement `QueryCache` with TTL and embedding-hash keying
- [ ] Implement `AdaptiveTopK` strategy in `query()` path
- [ ] Benchmark token savings on QA dataset: target 3000× average vs naive stuffing
- [ ] No regression in R@1 on standard benchmark

**Effort:** Low (2–3 days)
**Impact:** Medium-High — reduces LLM API bills for high-volume users

---

## Track 4: Async Batch Ingestion Pipeline

**Goal:** Production-grade throughput for data import.

**Current state:**
- `add_node` is synchronous, single-threaded
- 10K nodes = 0.4 s (exact) or 3.3 s (HNSW)
- 1M nodes would take 6–55 minutes of blocking time

**Target state:**
- **Batch insert API:** `field.add_nodes_batch(embeddings, contents, phases)` — vectorized
- **Background HNSW rebuild:** index updates queued, merged every N seconds
- **WAL (Write-Ahead Log):** every mutation appended to disk before memory commit
  - Crash recovery: replay WAL on startup
- **Target throughput:** 50K+ nodes/sec for batch, 1K+ nodes/sec for streaming

**Acceptance criteria:**
- [ ] Implement `add_nodes_batch()` with vectorized cache rebuild
- [ ] Implement `AsyncIndexBuilder` thread for HNSW background merge
- [ ] Implement `WALAppender` with fsync every 100 ms
- [ ] Benchmark: ingest 1M nodes in < 60 seconds

**Effort:** Medium-High (1 week)
**Impact:** High — required for production data pipelines

---

## Token Economics Reference (10K Nodes)

| Metric | Value |
|--------|-------|
| Corpus size | 10,000 documents |
| Avg tokens / document | ~17 |
| **Total corpus tokens** | **~170,000** |
| Naive stuffing (all docs to LLM) | 170,000 tokens / query |
| RTMDK retrieval (top-5) | ~85 tokens / query |
| **Token savings vs naive** | **2,000×** |
| Context window required (naive) | 170K+ (exceeds all current models) |
| Context window required (RTMDK) | 85 + system prompt (~500) = ~600 |
| **"Room to breathe" in 128K window** | **127,000 tokens for answer + reasoning** |

### Comparison with Ordinary RAG

| Dimension | Ordinary RAG (Chroma + SBERT) | RTMDK |
|-----------|-------------------------------|-------|
| Tokens sent to LLM | ~85 (same top-k) | ~85 (same top-k) |
| Embedding cost | $0.10–1.00 per 1M tokens | **$0** (SOT) |
| RAM footprint | 500 MB+ (model + DB) | **19–30 MB** |
| Latency | 50–500 ms + network | **1.3 ms** local |
| Offline capable | Only if SBERT local | **Yes, always** |
| Extra dependencies | torch, transformers, chroma | **numpy only** |

**Key insight:** RTMDK does not beat ordinary RAG on *token count to LLM* (both use top-k).
RTMDK wins on **total cost of ownership**, **latency**, **memory**, and **zero external dependencies**.

---

## Priority Matrix

| Track | Effort | Impact | Recommended Order |
|-------|--------|--------|-------------------|
| 1. Quantization | Low | Very High | **1st** |
| 3. Query Cache + Dynamic top_k | Low | Medium-High | **2nd** |
| 4. Async Batch Ingestion | Medium-High | High | **3rd** |
| 2. Tiered Storage | High | Very High | **4th** |

---

## How to Pick the Next Track

- **Shipping to mobile / edge devices** → Track 1 (Quantization)
- **High-volume chatbot, reducing OpenAI bills** → Track 3 (Query Cache)
- **Enterprise pilot with 1M+ document corpus** → Track 4 (Async Pipeline) then Track 2 (Tiered Storage)
- **Indie hacker, single-server deployment** → Track 3, then Track 1
