"""
PRODUCTION GUIDE: RTMDK Scaling Roadmap
================================================================================

> ⚠️ **IMPORTANT DISCLAIMER (2026-05-07):** This document describes a **roadmap**
> and **planned architecture**, not implemented features. As of v8.3, the following
> components do **NOT** exist in the codebase:
> - Product Quantization (PQ-64)
> - Distributed sharding with coordinator
> - Raft consensus and replication
> - Tiered routing or cold/hot separation
>
> Verified production capabilities (v8.3):
> - Single-machine deployment up to ~10K nodes (tested)
> - HNSW approximate search (optional, recommended for >5K nodes)
> - Async consolidation (non-blocking)
> - Query/embedding caching
> - Prometheus metrics and health monitoring
> - OpenAI-compatible API server
>
> The roadmap below is preserved for research planning and community contributions.

This document explains:
1. How to scale RTMDK from 100K to 10M+ nodes (planned)
2. Distributed architecture for production deployment (planned)
3. Performance tuning parameters (implemented)
4. Monitoring and alerting setup (partially implemented)

================================================================================
PART 1: SCALING TO 100K NODES (Single Machine)
================================================================================

Current limits (v8.3, tested on Windows 11, Python 3.10, Ryzen 5):
- N=1,000: 95.6% R@1, <1ms P95 latency, ~16MB RAM
- N=5,000: 100% R@1*, 1.4ms P95 latency, ~300MB RAM
- N=10,000: 100% R@1*, 1.9ms P99 latency, ~333MB RAM
- N=100,000: Needs 6 optimizations (see below) — not yet tested
- N=1,000,000: Requires distributed architecture (roadmap)

\* R@1 measured on synthetic semantic variants; real-world accuracy ~95%

The 6 Critical Optimizations for N > 100K:
────────────────────────────────────────────

1. TWO-STAGE RETRIEVAL
   Problem: O(N) resonance computation → 500ms at N=100K
   Solution: HNSW coarse filter (top-500) → resonance on 500 nodes
   Result: 500ms → 12ms

   Implementation:
   - Use HNSW with M=64, ef_construction=800 for better recall
   - Query HNSW for top-500 candidates → compute resonance only on those
   - This reduces O(N) to O(500) = constant time

2. VECTOR QUANTIZATION (PQ-64)
   Problem: 100K nodes × 3072 bytes = 307MB RAM
   Solution: Product Quantization → 64x compression
   Result: 307MB → 4.8MB

   Implementation:
   - Split 768D vector into 64 sub-vectors of 12D each
   - Cluster each sub-space into 256 centroids (1 byte per sub-vector)
   - Store only centroid indices: 64 bytes instead of 3072 bytes
   - Reconstruct on-the-fly during retrieval

3. APPROXIMATE CONSOLIDATION
   Problem: O(N²) consolidation → 50 seconds at N=100K
   Solution: K-Means clustering (k=1000) → consolidate within clusters only
   Result: 50s → 0.5s

   Implementation:
   - Run K-Means every 1000 steps (not every step)
   - Consolidate only nodes within the same cluster
   - Cross-cluster consolidation: sample 10 nodes per cluster pair

4. INCREMENTAL HNSW INDEX
   Problem: Index rebuild blocks inserts at 1000 inserts/sec
   Solution: Delta buffer + background merge
   Result: Zero-downtime inserts

   Implementation:
   - New nodes go into delta buffer (in-memory)
   - Background thread merges delta → main index every 5 minutes
   - Queries check both main index and delta buffer

5. BM25 OPTIMIZATION
   Problem: Vocabulary explosion → 500MB at N=100K
   Solution: Stemming + Stopword removal + Term pruning
   Result: 500MB → 50MB

   Implementation:
   - Apply Porter Stemmer to all terms
   - Remove stopwords (the, is, at, which, etc.)
   - Prune terms with document frequency < 2

6. GRAPH PRE-COMPUTATION
   Problem: Multi-hop requires 3-5 sequential retrievals
   Solution: Pre-compute top-3 neighbors for each node
   Result: 75ms → 8ms

   Implementation:
   - Offline: compute adjacency list for all nodes
   - Store as: node_id → [neighbor_id_1, neighbor_id_2, neighbor_id_3]
   - On query: lookup neighbors → retrieve their context → merge

================================================================================
PART 2: DISTRIBUTED ARCHITECTURE FOR N > 1M
================================================================================

Why Single-Machine Fails at 1M+
────────────────────────────────
- HNSW index: 15GB RAM at N=1M (even with PQ)
- Consolidation: K-Means on 1M nodes = 30 seconds
- Network I/O: 1M nodes × 2KB = 2GB per backup
- Single point of failure

The Solution: Distributed RTMDK Cluster
───────────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────┐
│                         LOAD BALANCER (nginx/HAPROXY)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  SHARD 0     │  │  SHARD 1     │  │  SHARD N     │                  │
│  │  0-333K      │  │  333K-666K   │  │  666K-1M     │                  │
│  │              │  │              │  │              │                  │
│  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │                  │
│  │  │HNSW    │  │  │  │HNSW    │  │  │  │HNSW    │  │                  │
│  │  │RTMDK   │  │  │  │RTMDK   │  │  │  │RTMDK   │  │                  │
│  │  │Memory  │  │  │  │Memory  │  │  │  │Memory  │  │                  │
│  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │                  │
│  │              │  │              │  │  │              │                  │
│  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │                  │
│  │  │PQ      │  │  │  │PQ      │  │  │  │PQ      │  │                  │
│  │  │Index   │  │  │  │Index   │  │  │  │Index   │  │                  │
│  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                          COORDINATOR (Redis/Raft)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐                │
│  │  Query      │  │  Consensus   │  │  Global HNSW    │                │
│  │  Router     │  │  Engine      │  │  Metadata       │                │
│  └─────────────┘  └──────────────┘  └─────────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘

Key Components:
───────────────

1. SHARDING STRATEGY
   - Hash-based: hash(query) % num_shards → route to shard
   - Consistent hashing: allows adding/removing shards without re-sharding
   - Each shard: independent RTMDKMemory with local HNSW index
   - Shard size: 300K-500K nodes (optimal for single machine)

2. QUERY ROUTING
   - Fan-out: send query to ALL shards → merge results
   - Fan-in: each shard returns top-K → coordinator picks global top-K
   - Latency: max(shard latency) + merge overhead (~5ms)
   - Example: 3 shards × 15ms = 15ms (parallel) + 2ms merge = 17ms total

3. CONSOLIDATION ACROSS SHARDS
   - Local consolidation: each shard consolidates independently (every 100 steps)
   - Global consolidation: once per hour, coordinator triggers cross-shard
   - Cross-shard: sample 100 representative nodes from each shard → compare
   - Merge conflicting nodes: keep higher salience, update causal links

4. REPLICATION & FAULT TOLERANCE
   - Each shard: 1 primary + 2 replicas (Raft consensus)
   - Writes go to primary → replicated to secondaries
   - Reads can go to any replica (eventual consistency)
   - Failover: if primary dies, replica promotes automatically

5. GLOBAL HNSW METADATA
   - Centroids: each shard computes K=100 centroids of its nodes
   - Coordinator: builds coarse HNSW over all centroids (3 shards × 100 = 300 centroids)
   - Query: check coarse HNSW → route to most relevant shard(s)
   - This avoids fan-out to all shards for simple queries

6. BACKUP & RECOVERY
   - Each shard: incremental backup every 5 minutes
   - Coordinator: global snapshot every hour
   - Recovery: restore shard from latest backup → replay WAL
   - WAL (Write-Ahead Log): all writes logged before applying

Configuration for 1M Nodes (3 shards):
───────────────────────────────────────

Shard Configuration (per shard):
  nodes_per_shard: 333,333
  pq_compression: 64 bytes per node
  ram_per_shard: ~50MB (nodes) + ~200MB (HNSW) = ~250MB
  total_ram_3_shards: ~750MB

Coordinator Configuration:
  centroids_per_shard: 100
  total_centroids: 300
  coarse_hnsw_ram: ~5MB
  coordinator_ram: ~50MB (metadata + routing tables)

Total Infrastructure:
  3 × Shard servers: 2 CPU, 512MB RAM each
  1 × Coordinator server: 1 CPU, 256MB RAM
  Total: 7 CPU, 1.8GB RAM

Expected Performance:
  Query latency (P95): 17ms (vs 500ms single-machine unoptimized)
  Insert throughput: 3000/sec (1000 per shard)
  Consolidation time: 1.5s local + 30s global (hourly)
  Backup size: 150MB per shard = 450MB total

================================================================================
PART 3: SCALING TO 10M+ NODES
================================================================================

For 10M+ nodes, the architecture scales horizontally:

10M Nodes = 30 shards × 333K nodes each
  - Query latency: ~20ms (30 shards in parallel)
  - RAM: 30 × 250MB = 7.5GB total
  - CPU: 30 × 2 = 60 cores total

100M Nodes = 300 shards × 333K nodes each
  - Query latency: ~25ms (300 shards in parallel)
  - RAM: 300 × 250MB = 75GB total
  - CPU: 300 × 2 = 600 cores total

Optimization for massive scale:
  - Tiered routing: coarse HNSW (1000 centroids) → mid-level → shard
  - Caching layer: Redis for frequent queries
  - Async writes: buffer inserts → batch to shards
  - Cold/hot separation: archive old nodes to object storage

================================================================================
PART 4: PRODUCTION MONITORING
================================================================================

Key Metrics to Monitor:
  1. Query latency (P50, P95, P99) — alert if P95 > 50ms
  2. Cache hit rate — alert if < 70%
  3. Memory usage per shard — alert if > 80% of allocated
  4. Consolidation time — alert if > 5s
  5. Replication lag — alert if > 10s
  6. Query throughput (QPS) — alert if drops below baseline
  7. Recall@1 (periodic benchmark) — alert if drops below 90%
  8. Node count per shard — alert if imbalanced (> 2x difference)

Alerting Setup (Prometheus + Grafana):
  - Scrape interval: 10s
  - Dashboard: shard health, query performance, memory usage
  - Alerts: PagerDuty for critical, Slack for warnings
  - Auto-scaling: add shard when node count > 400K

================================================================================
PART 5: MIGRATION PATH FROM SINGLE-MACHINE TO DISTRIBUTED
================================================================================

Step 1: Optimize single-machine (this implementation)
  - Implement 6 optimizations for N up to 100K
  - Monitor performance metrics

Step 2: Add sharding (2-3 weeks)
  - Split single RTMDKMemory into 2 shards
  - Add coordinator for query routing
  - Test with 200K nodes

Step 3: Add replication (1-2 weeks)
  - Add Raft consensus for each shard
  - Test failover scenarios
  - Deploy to staging

Step 4: Scale to 1M+ (ongoing)
  - Monitor metrics, add shards as needed
  - Optimize coordinator for large shard counts
  - Implement cold/hot separation for cost reduction

================================================================================
END OF PRODUCTION GUIDE
================================================================================
