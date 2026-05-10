# RTMDK Development Backlog

> Last updated: 2026-05-07
> Current version: 8.3.0-dev
> Test status: 956 passed, 1 skipped
> Branch: `refactor/leadership-cleanup`

---

## Executive Summary

RTMDK v8.3 is in active development. The **Leadership Cleanup** refactor (v8.3-alpha) has decoupled the monolithic `RTMDKField` (5265 → 844 lines, −84%) and `RTMDKMemory` (2603 → ~1380 lines, −47%) into 21 focused subsystems.

v8.2 remains production-ready for single-node deployment (10K–100K nodes, sub-10ms latency).
The backlog below targets **enterprise scale** (1M+ nodes), **cost reduction**, and **deployment ergonomics**.

---

## Completed in v8.2.1

- [x] **GraphQL API** — Strawberry schema at `/graphql` (Query + Mutation)
- [x] **WebSocket Streaming** — Real-time `/ws/memory` endpoint
- [x] **SOT Vocabulary Endpoint** — `/v1/sot/vocab` with pagination and search
- [x] **SOT Persistence** — SOT state included in `field_to_dict()` / `field_from_file()`
- [x] **SOT Graceful Degradation** — LRU eviction instead of RuntimeError on vocab full
- [x] **SOT Circuit Breaker** — Protects SBERT bootstrap from cascading failures
- [x] **React Admin Panel** — Vite + React dashboard (`admin/`)
- [x] **Vector-Native Storage Stub** — SQLite-VSS / pgvector migration path
- [x] **Multi-Master Replication Stub** — Raft/Paxos distributed consensus

## Completed in v8.3-alpha (Leadership Cleanup)

- [x] **FieldInitializer** — Extracted 460-line `RTMDKField.__init__` into `memory/field_initializer.py`
- [x] **ContextManager** — Extracted save/load context pipeline from `core.py` → `memory/context_manager.py`
- [x] **MemoryPostInitializer** — Extracted `model_post_init` backlog wiring → `memory/memory_post_initializer.py`
- [x] **BacklogModulesInitializer** — Extracted `_init_backlog_modules` → `memory/backlog_modules_initializer.py`
- [x] **PipelineBuilder** — Extracted `build_pipeline` → `memory/pipeline_builder.py`
- [x] **OperationalManager** — Moved `_compress_field`, `_self_heal`, calibrate, rollback into `memory/operational_manager.py`
- [x] **Dead code removal** — 23 lines of orphan imports, `_copy_node`, duplicate method definitions
- [x] **Import cycle fix** — `MemoryNode` import routed through `rtmdk.nodes` instead of `core.py`
- [x] **Backward compatibility** — All public APIs preserved via thin wrappers / `__getattr__`

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

## Track 2: Tiered Storage (Hot / Warm / Cold) — Partially Shipped

**Goal:** Support unlimited node count without proportional RAM growth.

**Current state:**
- All nodes live in RAM (`self.nodes` dict + numpy caches)
- 10K nodes = 30 MB; 1M nodes = ~3 GB — impractical on consumer hardware
- `TieredStorageManager` exists in `storage/tiered.py` (memmap + LFU stubs) but is **not wired** into `RTMDKField` hot path

**Target state:**
- **Hot** (top 1% by query frequency) → RAM
- **Warm** (next 9%) → memory-mapped disk (`numpy.memmap`)
- **Cold** (remaining 90%) → compressed msgpack snapshots
- Query path checks hot first, then warm, then cold (lazy load)

**Acceptance criteria:**
- [ ] Wire `TieredStorageManager` into `RTMDKField` query/add paths (replace `self.nodes` dict)
- [ ] Implement `TieredNodeStore` with hot/warm/cold tiers
- [ ] Benchmark: 1M nodes on 16 GB RAM with p99 query < 20 ms (deferred to v8.3 stress test)
- [x] Auto-promotion/demotion based on access frequency (LFU cache)
- [x] Graceful degradation: if cold node requested, load + promote to warm

**Effort:** High (1–2 weeks)
**Impact:** Very High — enterprise sales enabler

---

## Track 3: Query Cache + Dynamic top_k (Shipped)

**Goal:** Further reduce LLM token spend and latency for repetitive traffic.

**Current state (shipped):**
- `QueryCache` integrated into `RTMDKField.query()` — keyed by MD5(embedding + params)
- Cache hit returns instantly (<0.1 ms vs 1.3 ms compute)
- Cache auto-invalidates on `add_node` (clear all)
- `adaptive_top_k=True` reduces `top_k` based on confidence:
  - top-1 score ≥ 0.95 → return 1 node (saves 4× context tokens)
  - top-1 score 0.80–0.95 → return 3 nodes
  - top-1 score < 0.80 → return 5 nodes
- Config flags: `query_cache_size`, `query_cache_ttl`, `adaptive_top_k`

**Acceptance criteria:**
- [x] Implement `QueryCache` with TTL and embedding-hash keying
- [x] Implement `AdaptiveTopK` strategy in `query()` path
- [x] No regression in R@1 on standard benchmark
- [ ] Benchmark token savings on QA dataset: target 3000× average vs naive stuffing (deferred to analytics pipeline)

**Effort:** Low (2–3 days) — done in 1 session
**Impact:** Medium-High — reduces LLM API bills for high-volume users

---

## Track 4: Async Batch Ingestion Pipeline — Partially Shipped

**Goal:** Production-grade throughput for data import.

**Current state:**
- `add_node` is synchronous, single-threaded
- `add_nodes_batch()` exists in `NodeManager` but does **not** use vectorized cache rebuild (iterates Python loop)
- `AsyncIndexBuilder` exists in `memory/async_index.py` but is **not wired** into batch path
- `WAL` exists in `memory/wal.py` with periodic fsync (`wal_fsync_interval_ms`), auto-replay on startup
- 10K nodes = 0.4 s (exact) or 3.3 s (HNSW); 1M nodes would take 6–55 minutes of blocking time

**Target state:**
- **Batch insert API:** `field.add_nodes_batch(embeddings, contents, phases)` — fully vectorized numpy ops
- **Background HNSW rebuild:** index updates queued, merged every N seconds via `AsyncIndexBuilder`
- **WAL durability:** fsync every 100 ms (already supported, needs benchmarking)
- **Target throughput:** 50K+ nodes/sec for batch, 1K+ nodes/sec for streaming

**Acceptance criteria:**
- [x] Implement `add_nodes_batch()` API surface
- [x] Implement `AsyncIndexBuilder` thread class
- [x] Implement `WAL` with periodic fsync and crash recovery replay
- [ ] Vectorize `add_nodes_batch()` internals (replace Python loop with numpy batch ops)
- [ ] Wire `AsyncIndexBuilder` into `NodeManager.add_nodes_batch()` for background HNSW rebuild
- [ ] Benchmark: ingest 1M nodes in < 60 seconds

**Effort:** Medium (3–4 days) — foundations exist, needs integration + vectorization
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
| 0. Leadership Cleanup (v8.3-alpha) | Medium | Very High | **Done** ✅ |
| 1. Quantization | Low | Very High | **1st** |
| 3. Query Cache + Dynamic top_k | Low | Medium-High | **2nd** (Shipped) |
| 4. Async Batch Ingestion | Medium | High | **3rd** |
| 2. Tiered Storage | High | Very High | **4th** |

---

## How to Pick the Next Track

- **Shipping to mobile / edge devices** → Track 1 (Quantization)
- **High-volume chatbot, reducing OpenAI bills** → Track 3 (Query Cache — already shipped)
- **Enterprise pilot with 1M+ document corpus** → Track 4 (Async Pipeline) then Track 2 (Tiered Storage)
- **Indie hacker, single-server deployment** → Track 1 (Quantization) or Track 4 (Batch ingestion for faster setup)
- **Improving code maintainability / onboarding devs** → Update `docs/08_ARCHITECTURE.md` to reflect v8.3 decoupled architecture
