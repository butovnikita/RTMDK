"""Ingest comprehensive RTMDK project information into memory."""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rtmdk import RTMDKMemory, RTMDKConfig
from sentence_transformers import SentenceTransformer

MEMORY_DIR = Path.home() / ".rtmdk"
MEMORY_MSGPACK = MEMORY_DIR / "memory.msgpack"
MEMORY_JSON = MEMORY_DIR / "memory.json"

# ------------------------------------------------------------------
# Embedder: local sentence-transformers (384-dim, RTMDK will project to latent_dim)
# ------------------------------------------------------------------
print("Loading embedder...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def embedder(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

# ------------------------------------------------------------------
# Project knowledge nodes
# ------------------------------------------------------------------
NODES = [
    {
        "title": "RTMDK Project Overview",
        "text": (
            "RTMDK (Resonance-Topological Memory) v8.3.0 is a Python 3.10 memory system for LLMs. "
            "Primary development on Windows, production deployment via Linux Docker. "
            "GPU support: CUDA 12.8, torch 2.12.0.dev. "
            "Project root: C:\\Users\\Никита\\Desktop\\llm_lab. "
            "Core purpose: provide resonant-topological memory storage, retrieval, and context injection for AI agents."
        ),
        "tags": ["overview", "project", "meta"],
    },
    {
        "title": "RTMDK Directory Structure",
        "text": (
            "Key directories: rtmdk/memory/ — core memory (core.py, field.py, config.py); "
            "rtmdk/memory/sot_v2/ — SOT v2.0 embedder (SIF + contrastive fine-tuning); "
            "rtmdk/production/ — production features (reranker, cascade router, BM25); "
            "rtmdk/engines/ — causal traversal, planners, evaluators; "
            "rtmdk/support/ — HNSW, projection, BM25, circuit breaker; "
            "rtmdk/server/ — FastAPI/GraphQL/WebSocket server; "
            "rtmdk/utils/ — JSON logging, async embedder; "
            "tests/ — 1112 pytest tests; scripts/ — benchmarks and diagnostics; docs/ — architecture and guides."
        ),
        "tags": ["architecture", "directories", "structure"],
    },
    {
        "title": "SOT v2.0 Embedder",
        "text": (
            "SOT v2.0 (Sense-Of-Text) is the native RTMDK embedder using Smooth Inverse Frequency (SIF) "
            "plus contrastive fine-tuning. Config key: sot_v2_enabled. "
            "Critical constraint: dense PMI matrix in sif_embedder.fit() scales as O(n_valid^2). "
            "For vocab > 5000, sparse PMI path (scipy.sparse + TruncatedSVD) activates automatically to prevent OOM. "
            "After any embedder training (train_sot_v2), the conformal calibrator MUST be reset or coverage guarantees are void."
        ),
        "tags": ["sot_v2", "embedder", "SIF", "PMI", "conformal"],
    },
    {
        "title": "Engram Embedding Cache",
        "text": (
            "EngramEmbeddingCache (rtmdk/memory/engram_cache.py) provides hot/warm/cold tiered cache "
            "to avoid TieredNodeStore disk scans during engram retrieval. "
            "Config: sot.engram_cache_enabled (default True), sot.engram_cache_max_hot (10k), sot.engram_cache_max_warm (90k). "
            "API: .add(nid, emb), .get(nid), .get_all(), .clear(), .save(path), .load(path). "
            "Wired into add_node() and _retrieve_and_format() when engram_manager is active. Thread-safe via RLock."
        ),
        "tags": ["backlog", "engram_cache", "cache", "tiered"],
    },
    {
        "title": "Observability and Telemetry",
        "text": (
            "Observability module (rtmdk/memory/observability.py) provides in-memory latency histograms (p50/p95/p99), "
            "cache hit/miss ratio, threshold alerting, and Prometheus export. "
            "Config: sot.observability_enabled (default False). "
            "Alert handlers: WebhookAlertHandler, SlackAlertHandler, PagerDutyAlertHandler. "
            "API: MemoryMetrics.record_query(latency_ms, cache_hit), .to_prometheus(), .flush_to_file(path), .check_alerts(). "
            "Wired into retrieve_nodes() for query latency and add_node() for ingestion latency."
        ),
        "tags": ["backlog", "observability", "telemetry", "metrics", "prometheus"],
    },
    {
        "title": "Distributed Lock",
        "text": (
            "DistributedLock (rtmdk/memory/distributed_lock.py) provides file-based or Redis inter-process lock "
            "with intra-process thread safety. "
            "Config: sot.distributed_lock_path, sot.distributed_lock_backend (file or redis), sot.distributed_lock_redis_url. "
            "API: .acquire(blocking=True), .release(), context-manager compatible. "
            "Wired into retrieve_nodes() to acquire/release around the retrieval pipeline."
        ),
        "tags": ["backlog", "distributed_lock", "redis", "concurrency"],
    },
    {
        "title": "RAG Quality Modules",
        "text": (
            "RAG Quality (rtmdk/memory/rag_quality.py) includes QueryDecomposer, SentenceReranker, and FeedbackLoop. "
            "QueryDecomposer.decompose(query) -> list of sub-queries. "
            "SentenceReranker.rerank(query, results, top_k) uses batched sentence embedding. "
            "FeedbackLoop.add_feedback(query, node_text, relevant) requires SOTv2 embedder. "
            "Config flags: sentence_reranker_enabled, query_decomposition_enabled (optional llm_client), feedback_loop_enabled, feedback_loop_persist_path. "
            "Wired into _retrieve_and_format() and retrieve_nodes(). Accessible via RTMDKMemory.add_feedback()."
        ),
        "tags": ["backlog", "rag", "reranker", "feedback", "quality"],
    },
    {
        "title": "Explainability",
        "text": (
            "Explainability (rtmdk/memory/explainability.py) provides ResultExplainer, QueryRewriter, and QueryIntentClassifier. "
            "ResultExplainer gives human-readable explanations for retrieved memories. "
            "QueryRewriter auto-rewrites queries when top-score < threshold. "
            "QueryIntentClassifier classifies intent as factual/exploratory/conversational/comparative. "
            "Config: result_explainability_enabled, query_rewrite_enabled, query_rewrite_threshold, query_intent_classification_enabled. "
            "API: retrieve_nodes_with_explanations(query, embedding, ...) -> {results, explanations, intent}."
        ),
        "tags": ["backlog", "explainability", "intent", "query_rewrite"],
    },
    {
        "title": "Safety Modules",
        "text": (
            "Safety (rtmdk/memory/safety.py) includes RollbackManager and PoisonedMemoryDetector. "
            "RollbackManager captures snapshots and rolls back memory to a previous state. "
            "PoisonedMemoryDetector detects nodes with high out-degree, repetitive content, or extreme sentiment. "
            "API: memory.take_snapshot(), memory.rollback(timestamp), memory.detect_poisoned_memories()."
        ),
        "tags": ["backlog", "safety", "rollback", "poisoned_memory"],
    },
    {
        "title": "Timeline and Narrative",
        "text": (
            "Timeline and Narrative (rtmdk/memory/timeline.py) includes MemoryTimeline and MemoryNarrator. "
            "MemoryTimeline provides chronological views of memories per session/tier/time range. "
            "MemoryNarrator generates human-readable stories from episodic memories with Markdown export. "
            "API: timeline.get_timeline(session_id=...), narrator.narrate_session(session_id), narrator.export_markdown(path)."
        ),
        "tags": ["backlog", "timeline", "narrative", "episodic"],
    },
    {
        "title": "Production Modules",
        "text": (
            "Production modules enabled by default: QueryCache, EmbeddingCache, ContextOptimizer, HealthMonitor, AuditLog. "
            "Located in rtmdk/production/. Features include reranker, cascade router, and BM25 sparse retrieval. "
            "Server configuration preset: production. Parameters: latent_dim=256, decay=0.999, tension=0.15, top_k=5."
        ),
        "tags": ["production", "modules", "query_cache", "health_monitor"],
    },
    {
        "title": "Server API Endpoints",
        "text": (
            "RTMDK Production API exposes: POST /v1/chat/completions (chat with memory), POST /v1/embeddings, "
            "POST /v1/memory/query, POST /v1/memory/batch_query, GET /v1/models, GET /health, GET /metrics (Prometheus), "
            "GET /dashboard (Web UI), GET /api/models (UX model selector), POST /api/config (runtime config), "
            "GET /v1/analytics/overview, GET /v1/analytics/memory, GET /v1/analytics/events, GET /v1/analytics/report. "
            "OpenAI-compatible. Default port 8080 (currently configured to 8081). API Key: rtmdk-local. "
            "LM Studio / OpenRouter URL: https://openrouter.ai/api/v1."
        ),
        "tags": ["server", "api", "endpoints", "fastapi"],
    },
    {
        "title": "JSON Logging and Async Embedder",
        "text": (
            "rtmdk/utils/json_logger.py provides structured JSON logging for ELK/Loki integration. "
            "Usage: setup_json_logging(level=logging.INFO). "
            "rtmdk/utils/async_embedder.py provides async wrapper with request batching for high-throughput scenarios. "
            "Usage: AsyncEmbedder(embed_fn, batch_size=16)."
        ),
        "tags": ["utils", "json_logging", "async_embedder", "elk", "loki"],
    },
    {
        "title": "State Persistence and Health Check",
        "text": (
            "RTMDKMemory.save_state(dir_path) persists engram cache, feedback buffer, and metrics snapshot. "
            "RTMDKMemory.load_state(dir_path) restores engram cache and feedback loop. "
            "RTMDKMemory.health_check() returns {status: healthy|degraded|unhealthy, checks: {...}}. "
            "Health checks cover: node count, memory, disk, circuit breaker state, metrics snapshot. "
            "Config validation: RTMDKConfig.validate() returns warning strings for conflicting settings; auto-called on init."
        ),
        "tags": ["persistence", "health_check", "config_validation"],
    },
    {
        "title": "Circuit Breaker for Embedder",
        "text": (
            "CircuitBreaker from rtmdk/support/circuit_breaker.py wraps the embedder. "
            "After 3 consecutive failures it returns a zero vector and logs a warning. "
            "Config: embedder_circuit_breaker_enabled (default True), embedder_cb_threshold, embedder_cb_recovery."
        ),
        "tags": ["circuit_breaker", "resilience", "embedder"],
    },
    {
        "title": "Critical Constraints Summary",
        "text": (
            "1. SIF OOM boundary: dense PMI matrix in sif_embedder.fit() scales O(n_valid^2); sparse path auto-activates for vocab > 5000. "
            "2. Conformal invalidation: after any embedder training, the conformal calibrator MUST be reset. "
            "3. Thread safety: add_node() uses threading.Lock for SOT online update buffer; EngramEmbeddingCache uses RLock. "
            "4. Config access: always use flat-field access (e.g., config.engram_cache_enabled) backed by _FIELD_GROUPS mapping."
        ),
        "tags": ["constraints", "thread_safety", "OOM", "conformal"],
    },
    {
        "title": "Testing and Benchmarks",
        "text": (
            "Full test suite: python -m pytest tests/ -x -q (1112 tests). "
            "Backlog modules tests: python -m pytest tests/test_backlog_modules.py -v. "
            "Benchmark: python scripts/bench_backlog_modules.py. "
            "Additional scripts in scripts/: bench_adaptive_ab.py, bench_adaptive_sbert.py, bench_beir_grid.py, bench_meta_adaptive.py, and 38 more."
        ),
        "tags": ["testing", "pytest", "benchmarks", "ci"],
    },
    {
        "title": "Git Workflow and Commits",
        "text": (
            "Commits use conventional format: feat(scope): description, fix(scope): description, refactor(scope): description. "
            "Push to main after all tests pass. Project uses GitHub with ISSUE_TEMPLATE, PULL_REQUEST_TEMPLATE, dependabot, and CI workflows."
        ),
        "tags": ["git", "workflow", "conventional_commits"],
    },
]


def main():
    cfg = RTMDKConfig.production()
    memory = RTMDKMemory(config=cfg, embedder=embedder)
    print("Created fresh memory with production config.")

    texts = [n["text"] for n in NODES]
    contents = [
        {
            "text": n["text"],
            "title": n["title"],
            "tags": n["tags"],
            "source": "project_ingestion",
            "ingested_at": time.time(),
        }
        for n in NODES
    ]

    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)

    print(f"Adding {len(texts)} nodes to memory...")
    nids = memory.add_nodes_batch(embeddings, contents)
    print(f"Added node IDs: {nids}")

    # Save to msgpack (primary)
    print(f"Saving to {MEMORY_MSGPACK} ...")
    memory.field.export_field(str(MEMORY_MSGPACK), fmt="msgpack")

    # Save to json (for server compatibility)
    print(f"Saving to {MEMORY_JSON} ...")
    memory.field.export_field(str(MEMORY_JSON), fmt="json")

    print("Done.")
    print(f"Total nodes in memory: {len(memory.field.nodes)}")
    memory.close()


if __name__ == "__main__":
    main()
