"""RTMDK Production Server — OpenAI-compatible API with Resonance-Topological Memory.

This is the CLEAN production version WITHOUT SillyTavern modules.
For development with SillyTavern support, use rtmdk_server.py instead.

Usage:
    python -m rtmdk
    python start_production.py
"""

from rtmdk.utils.lru_cache import LRUCache
import signal
import atexit
import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

# RTMDK package imports
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory

from rtmdk.production.query_cache import QueryCache
from rtmdk.production.embedding_cache import EmbeddingCache
from rtmdk.production.context_optimizer import ContextOptimizer
from rtmdk.production.health_monitor import HealthMonitor
from rtmdk.production.analytics_dashboard import AnalyticsDashboard
from rtmdk.production.api_key_manager import APIKeyManager
from rtmdk.production.tenant_rate_limiter import TenantRateLimiter
from rtmdk.production.webhooks import WebhookManager
from rtmdk.production.retention import RetentionManager, RetentionPolicy
from rtmdk.production.audit_log import AuditLog
from rtmdk.production.redis_cache import RedisQueryCache, RedisEmbeddingCache
from rtmdk.production.encryption import EncryptionManager
from rtmdk.production.telemetry import TelemetryManager
from rtmdk.server.grpc_service import serve_grpc
from rtmdk.server.graphql_schema import schema
from rtmdk.support.circuit_breaker import AsyncCircuitBreaker

# GraphQL
_sot_bootstrap_breaker = AsyncCircuitBreaker("SOTBootstrap", failure_threshold=3, recovery_timeout=60.0)

try:
    from strawberry.fastapi import GraphQLRouter
    graphql_router = GraphQLRouter(schema)
    GRAPHQL_AVAILABLE = True
except Exception:
    graphql_router = None  # type: ignore[assignment]
    GRAPHQL_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

if _PROMETHEUS_AVAILABLE:
    _metric_nodes = Gauge("rtmdk_nodes_total", "Number of memory nodes")
    _metric_queries = Counter("rtmdk_queries_total", "Total memory queries")
    _metric_query_dur = Histogram(
        "rtmdk_query_duration_seconds",
        "Query duration")
    _metric_consolidations = Counter(
        "rtmdk_consolidations_total",
        "Total consolidations")
    _metric_security = Counter(
        "rtmdk_security_violations_total",
        "Security violations")
    _metric_lm_requests = Counter(
        "rtmdk_lm_requests_total",
        "LM Studio requests",
        ["endpoint"])
    _metric_lm_errors = Counter(
        "rtmdk_lm_errors_total",
        "LM Studio errors",
        ["endpoint"])
    # SOT-specific metrics
    _metric_sot_vocab = Gauge(
        "rtmdk_sot_vocab_size",
        "SOT vocabulary size",
        ["mode"])
    _metric_sot_cooccurrence = Gauge(
        "rtmdk_sot_cooccurrence_size",
        "SOT cooccurrence store entries")
    _metric_sot_bootstrap_time = Gauge(
        "rtmdk_sot_bootstrap_time_seconds",
        "SOT bootstrap duration")


# ============================================================================
# ASYNC HELPERS
# ============================================================================


async def run_sync(func, *args, **kwargs):
    """Run a synchronous function in the default thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ============================================================================
# CONFIGURATION
# ============================================================================

SERVER_HOST = os.getenv("RTMDK_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("RTMDK_PORT", "8080"))
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:12345/v1")
MEMORY_FILE = os.getenv(
    "RTMDK_MEMORY_FILE",
    os.path.join(
        os.path.expanduser("~"),
        ".rtmdk",
        "memory.json"))
EMBED_MODEL = os.getenv(
    "RTMDK_EMBED_MODEL",
    "nomic-ai/nomic-embed-text-v1.5-GGUF")
API_KEY = os.getenv("RTMDK_API_KEY", "rtmdk-local")
ENABLE_LM_STUDIO = os.getenv(
    "RTMDK_ENABLE_LM_STUDIO",
    "true").lower() == "true"
ENABLE_API_AUTH = os.getenv("RTMDK_ENABLE_API_AUTH", "true").lower() == "true"
MAX_PAYLOAD_SIZE = int(os.getenv("RTMDK_MAX_PAYLOAD_SIZE", "1048576"))
ALLOWED_ORIGINS = os.getenv("RTMDK_ALLOWED_ORIGINS", "*").split(",")

# Shared async HTTP client
http_client = httpx.AsyncClient()

# Graceful shutdown state
_active_requests = 0
_shutdown_event = asyncio.Event()

# Pipeline metrics store (optional)
pipeline_metrics_store = None


async def _sot_bootstrap_from_memory():
    """Bootstrap SOT tokenizer from existing memory nodes in background."""
    await asyncio.sleep(2)  # Let server finish startup
    if memory is None or memory.field is None or memory.field.sot_tokenizer is None:
        return
    try:
        texts = []
        for node in memory.field.nodes.values():
            text = node.content.get("text", "") if isinstance(node.content, dict) else str(node.content)
            if text:
                texts.append(text)
        if len(texts) >= 10:
            memory.field.sot_tokenizer.warm_start_from_corpus(texts)
            logger.info(f"SOT bootstrapped from {len(texts)} memory nodes")
    except Exception:
        logger.debug("SOT background bootstrap failed", exc_info=True)


async def _drain_active_requests(timeout: float = 30.0):
    """Wait for active requests to complete before shutdown."""
    t0 = time.time()
    while _active_requests > 0 and (time.time() - t0) < timeout:
        logger.info(f"Draining {_active_requests} active requests...")
        await asyncio.sleep(0.5)
    if _active_requests > 0:
        logger.warning(f"Shutdown with {_active_requests} requests still active")
    else:
        logger.info("All requests drained gracefully")

# ============================================================================
# SIGNAL / LIFECYCLE HANDLERS
# ============================================================================


_memory_ref = None  # set in startup_event


def _handle_sigterm(signum, frame):
    logger.info("Received SIGTERM, initiating graceful shutdown...")
    if _memory_ref is not None:
        try:
            save_path = _get_save_path(MEMORY_FILE)
            _memory_ref.export_field(save_path)
            logger.info(f"Memory saved to {save_path} on SIGTERM")
        except Exception:
            logger.exception("Failed to save memory on SIGTERM")
    # Allow default handler to terminate the process
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)


@atexit.register
def _atexit_save():
    if _memory_ref is not None:
        try:
            save_path = _get_save_path(MEMORY_FILE)
            _memory_ref.export_field(save_path)
            logger.info(f"Memory saved to {save_path} at exit")
        except Exception:
            logger.exception("Failed to save memory at exit")


# ============================================================================
# LOGGING
# ============================================================================


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for production observability."""

    def format(self, record):
        obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            obj["request_id"] = record.request_id
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


log_level = getattr(logging, os.getenv("RTMDK_LOG_LEVEL", "INFO"))
_log_handler = logging.StreamHandler()
if os.getenv("RTMDK_LOG_FORMAT", "").lower() == "json":
    _log_handler.setFormatter(JsonFormatter())
else:
    _log_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.basicConfig(level=log_level, handlers=[_log_handler])
logger = logging.getLogger("rtmdk_production")

# ============================================================================
# APP INITIALIZATION
# ============================================================================


async def _health_check_loop():
    """Background task that runs health checks periodically."""
    while True:
        await asyncio.sleep(60)
        if health_monitor:
            health_monitor.check_health()


async def _auto_save_loop():
    """Background task that auto-saves memory periodically."""
    interval = int(os.getenv("RTMDK_AUTO_SAVE_INTERVAL", "60"))
    while True:
        await asyncio.sleep(interval)
        auto_save()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global memory, lm_studio_available
    global query_cache, embedding_cache, context_optimizer, health_monitor
    global analytics_dashboard, api_key_manager, tenant_rate_limiter
    global webhook_manager, audit_log, retention_manager
    global redis_query_cache, redis_embedding_cache, encryption_manager, telemetry_manager
    # Structured JSON logging in production mode
    if os.getenv("RTMDK_JSON_LOG", "").lower() in ("1", "true", "yes"):
        try:
            from rtmdk.production.json_logger import setup_json_logging
            setup_json_logging()
        except Exception:
            pass
    logger.info("Starting RTMDK Production API v8.2.0")
    logger.info(f"Memory file: {MEMORY_FILE}")
    logger.info(f"LM Studio URL: {LM_STUDIO_URL}")

    if ENABLE_LM_STUDIO:
        lm_studio_available = await check_lm_studio()

    memory = init_memory()
    global _memory_ref
    _memory_ref = memory

    # Initialize production performance modules
    query_cache = QueryCache(max_size=10000, ttl_seconds=3600)
    embedding_cache = EmbeddingCache(
        cache_dir=os.path.join(
            os.path.expanduser("~"),
            ".rtmdk",
            "embedding_cache"),
        max_size=100000,
        ttl_seconds=86400,
        memory_cache_size=4096,
    )
    context_optimizer = ContextOptimizer(
        model="default", min_tokens=50, max_tokens=300)
    health_monitor = HealthMonitor(memory=memory, check_interval=60)
    analytics_dashboard = AnalyticsDashboard(memory, health_monitor=health_monitor)
    api_key_manager = APIKeyManager()
    tenant_rate_limiter = TenantRateLimiter(api_key_manager=api_key_manager)
    webhook_manager = WebhookManager()
    audit_log = AuditLog()

    # Retention manager
    retention_manager = RetentionManager(memory.field)
    retention_manager.set_policy(
        RetentionPolicy(
            max_age_seconds=float(os.getenv("RTMDK_RETENTION_MAX_AGE_DAYS", "0")) * 86400 or None,
            max_nodes=int(os.getenv("RTMDK_RETENTION_MAX_NODES", "0")) or None,
        )
    )
    retention_manager.start()

    # Redis caches
    redis_query_cache = RedisQueryCache(redis_url=os.getenv("REDIS_URL"))
    redis_embedding_cache = RedisEmbeddingCache(redis_url=os.getenv("REDIS_URL"))
    if redis_query_cache.available:
        logger.info("Redis query cache connected")
    if redis_embedding_cache.available:
        logger.info("Redis embedding cache connected")

    # Encryption at rest
    encryption_manager = EncryptionManager()
    if encryption_manager.enabled:
        logger.info("Encryption at rest enabled")

    # OpenTelemetry tracing
    telemetry_manager = TelemetryManager()
    if telemetry_manager.enabled:
        logger.info("OpenTelemetry tracing enabled")

    # SOT checkpoint loading
    _sot_checkpoint_path = os.path.join(
        os.path.expanduser("~"), ".rtmdk", "sot_checkpoint.json"
    )
    if memory and memory.field and memory.field.sot_tokenizer:
        if os.path.exists(_sot_checkpoint_path):
            try:
                with open(_sot_checkpoint_path, "r", encoding="utf-8") as fh:
                    sot_state = json.load(fh)
                memory.field.sot_tokenizer.load_state(sot_state)
                logger.info(f"SOT checkpoint loaded ({len(sot_state.get('token_embeddings', {}))} tokens)")
            except Exception:
                logger.warning("Failed to load SOT checkpoint", exc_info=True)
        asyncio.create_task(_sot_bootstrap_from_memory())

    asyncio.create_task(_health_check_loop())
    asyncio.create_task(_auto_save_loop())

    # gRPC server (optional)
    grpc_port = int(os.getenv("RTMDK_GRPC_PORT", "0"))
    if grpc_port > 0:
        asyncio.create_task(serve_grpc(port=grpc_port))
        logger.info(f"gRPC server started on port {grpc_port}")

    logger.info(f"Server ready on {SERVER_HOST}:{SERVER_PORT}")
    if memory.field is not None:
        logger.info(f"Memory nodes: {len(memory.field.nodes)}")
    logger.info(
        "Production modules: QueryCache, EmbeddingCache, ContextOptimizer, HealthMonitor, AuditLog enabled")

    yield

    logger.info("RTMDK server shutting down...")
    _shutdown_event.set()
    await _drain_active_requests(timeout=30.0)
    if telemetry_manager is not None:
        telemetry_manager.shutdown()
    if retention_manager is not None:
        retention_manager.stop()
    if audit_log is not None:
        audit_log.close()
    if memory:
        # Gracefully stop background workers
        field = memory.field
        if field is not None:
            for task in field._workers:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=10.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
            field._workers.clear()

        # SOT checkpoint saving
        if field and field.sot_tokenizer:
            try:
                sot_state = field.sot_tokenizer.get_state()
                _sot_checkpoint_path = os.path.join(
                    os.path.expanduser("~"), ".rtmdk", "sot_checkpoint.json")
                os.makedirs(os.path.dirname(_sot_checkpoint_path), exist_ok=True)
                with open(_sot_checkpoint_path, "w", encoding="utf-8") as fh:
                    json.dump(sot_state, fh, ensure_ascii=False, default=str)
                logger.info(f"SOT checkpoint saved ({len(sot_state.get('token_embeddings', {}))} tokens)")
            except Exception:
                logger.exception("Failed to save SOT checkpoint")

        try:
            save_path = _get_save_path(MEMORY_FILE)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            await run_sync(memory.export_field, save_path)
            if memory.field is not None:
                logger.info(
                    f"Memory saved to {save_path} ({len(memory.field.nodes)} nodes)")
        except Exception:
            logger.exception("Failed to save memory on shutdown")


app = FastAPI(
    title="RTMDK Production API",
    description="OpenAI-compatible API with Resonance-Topological Memory (No SillyTavern)",
    version="8.2.0",
    lifespan=lifespan,
)

if GRAPHQL_AVAILABLE and graphql_router is not None:
    app.include_router(graphql_router, prefix="/graphql")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from rtmdk.production.metrics import MetricsMiddleware
    app.add_middleware(MetricsMiddleware)
except Exception:
    pass


# ============================================================================
# REQUEST COUNTER MIDDLEWARE (for graceful shutdown draining)
# ============================================================================


@app.middleware("http")
async def request_counter_middleware(request: Request, call_next):
    """Count active requests for graceful shutdown."""
    global _active_requests
    if _shutdown_event.is_set():
        return JSONResponse(
            status_code=503,
            content={"error": "Server is shutting down"})
    _active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        _active_requests -= 1


# ============================================================================
# REQUEST TIMEOUT MIDDLEWARE
# ============================================================================


@app.middleware("http")
async def request_timeout_middleware(request: Request, call_next):
    """Enforce per-request timeout."""
    timeout = float(os.getenv("RTMDK_REQUEST_TIMEOUT", "60"))
    try:
        return await asyncio.wait_for(call_next(request), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("Request timeout: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=504,
            content={"error": "Request timeout"})


# ============================================================================
# STRUCTURED REQUEST LOGGING MIDDLEWARE
# ============================================================================


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with structured JSON for production observability."""
    import uuid
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    t0 = time.time()
    try:
        response = await call_next(request)
        latency_ms = round((time.time() - t0) * 1000, 2)
        logger.info(
            json.dumps({
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "tenant_id": getattr(request.state, "tenant_id", None),
                "client_host": request.client.host if request.client else None,
            }, ensure_ascii=False)
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        latency_ms = round((time.time() - t0) * 1000, 2)
        logger.warning(
            json.dumps({
                "event": "http_request_error",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "error": str(exc),
                "latency_ms": latency_ms,
                "tenant_id": getattr(request.state, "tenant_id", None),
            }, ensure_ascii=False)
        )
        raise


# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enforce API key authentication, resolve tenant, and check payload limits."""
    skip_auth_paths = {
        "/health",
        "/v1/models",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/dashboard"}
    if request.url.path in skip_auth_paths or request.url.path.startswith("/api/"):
        return await call_next(request)

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
        return JSONResponse(
            status_code=413, content={"error": "Payload too large"})

    if ENABLE_API_AUTH:
        auth_header = request.headers.get("authorization", "")
        api_key = auth_header.replace("Bearer ", "").replace("bearer ", "") if auth_header else ""
        if not api_key:
            api_key = request.headers.get("x-api-key", "")

        tenant_id = None
        if api_key_manager is not None:
            tenant_id = api_key_manager.validate_key(api_key)

        # Fallback to legacy global API_KEY for backward compatibility
        if tenant_id is None and api_key == API_KEY:
            tenant_id = "__admin__"

        if tenant_id is None:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized. Provide valid API key."})

        request.state.tenant_id = tenant_id
        request.state.api_key = api_key

    return await call_next(request)


# Phase 4: Tenant-aware rate limiter middleware
# Note: old global rate_limiter replaced by tenant_rate_limiter initialized in lifespan


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-tenant rate limits."""
    skip_paths = {
        "/health",
        "/v1/models",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/dashboard",
        "/metrics"}
    if request.url.path in skip_paths or request.url.path.startswith("/api/"):
        return await call_next(request)

    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id is None:
        # If auth disabled, rate-limit by IP
        tenant_id = request.client.host if request.client else "anonymous"

    if tenant_rate_limiter is not None:
        is_pipeline = request.url.path.startswith("/v1/memory/pipeline/")
        allowed = (
            tenant_rate_limiter.allow_pipeline_request(tenant_id)
            if is_pipeline
            else tenant_rate_limiter.allow_request(tenant_id)
        )
        if not allowed:
            remaining = tenant_rate_limiter.get_remaining(tenant_id)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "remaining": remaining})

    return await call_next(request)


# Phase 4: Circuit breakers for external calls
llm_chat_circuit = AsyncCircuitBreaker(
    "llm_chat",
    failure_threshold=3,
    recovery_timeout=30.0,
    default=None)
llm_embed_circuit = AsyncCircuitBreaker(
    "llm_embed",
    failure_threshold=3,
    recovery_timeout=30.0,
    default=None)


# ============================================================================
# GLOBAL STATE
# ============================================================================

memory: Optional[RTMDKMemory] = None

embedder_cache = LRUCache(maxsize=4096)
lm_studio_available: bool = False
chat_model: Optional[str] = None


# ============================================================================
# REQUEST MODELS
# ============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "rtmdk",
                "messages": [{"role": "user", "content": "What do I know about coffee?"}],
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False,
                "session_id": "default",
            }
        }
    )

    model: str = "rtmdk"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    session_id: str = "default"


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "rtmdk-embed",
                "input": "The quick brown fox jumps over the lazy dog.",
            }
        }
    )

    model: str = "rtmdk-embed"
    input: str | List[str]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def check_lm_studio() -> bool:
    """Check if LM Studio is available."""
    try:
        resp = await http_client.get(f"{LM_STUDIO_URL}/models", timeout=3)
        global chat_model
        models = resp.json().get("data", [])
        if models:
            chat_model = models[0]["id"]
            logger.info(f"LM Studio detected: {chat_model}")
            return True
    except Exception:
        logger.warning(
            "LM Studio not available at %s",
            LM_STUDIO_URL,
            exc_info=True)
    return False


async def _fetch_embedding(text: str, model: str) -> np.ndarray:
    """Raw embedding fetch — wrapped by circuit breaker."""
    if _PROMETHEUS_AVAILABLE:
        _metric_lm_requests.labels(endpoint="embeddings").inc()
    resp = await http_client.post(
        f"{LM_STUDIO_URL}/embeddings",
        json={"model": model, "input": text},
        timeout=30,
    )
    data = resp.json()
    return np.array(data["data"][0]["embedding"], dtype=np.float32)


async def _get_embedding_cached(text: str, model: str = None) -> np.ndarray:
    """Get embedding with disk+memory caching."""
    if embedding_cache is None:
        return await get_embedding(text, model)

    # Check caches synchronously first
    key = embedding_cache._make_key(text)
    if key in embedding_cache.memory_cache:
        emb, timestamp = embedding_cache.memory_cache[key]
        if time.time() - timestamp < embedding_cache.ttl:
            embedding_cache._hits += 1
            embedding_cache.memory_cache.move_to_end(key)
            return emb
        else:
            del embedding_cache.memory_cache[key]

    cached_emb = embedding_cache._load_from_disk(key)
    if cached_emb is not None:
        embedding_cache._hits += 1
        embedding_cache._save_to_memory_cache(key, cached_emb)
        return cached_emb

    # Miss — compute async
    embedding_cache._misses += 1
    emb = await get_embedding(text, model)
    embedding_cache._save_to_memory_cache(key, emb)
    embedding_cache._save_to_disk(key, emb)
    return emb


async def get_embedding(text: str, model: str = None) -> np.ndarray:
    """Get embedding from SOT (primary), LM Studio (fallback), or cache."""
    cached = embedder_cache.get(text)
    if cached is not None:
        return cached

    # Phase 21: SOT primary embedder — works out-of-the-box without LM Studio
    if memory and memory.field and memory.field.sot_tokenizer:
        try:
            sot = memory.field.sot_tokenizer
            tokens = sot.encode(text)
            emb = sot.embed(tokens)
            embedder_cache.set(text, emb)
            return emb
        except Exception:
            logger.debug("SOT embedding failed for: %s...", text[:50])

    # Fallback to LM Studio / external embedder
    embedder_model = model or EMBED_MODEL
    embedding = await llm_embed_circuit.call(_fetch_embedding, text, embedder_model)
    if embedding is None:
        if _PROMETHEUS_AVAILABLE:
            _metric_lm_errors.labels(endpoint="embeddings").inc()
        logger.warning("Embedding fallback for: %s...", text[:50])
        # Deterministic fallback: stable for same text, low amplitude to
        # minimize field distortion
        rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
        emb = rng.standard_normal(768).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8) * 0.01
        embedder_cache.set(text, emb)
        return emb

    expected_dim = 768
    if len(embedding) != expected_dim:
        logger.warning(
            f"Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}. Resizing.")
        if len(embedding) > expected_dim:
            embedding = embedding[:expected_dim]
        else:
            embedding = np.pad(
                embedding, (0, expected_dim - len(embedding)), "constant")

    embedder_cache.set(text, embedding)
    return embedding


def init_memory() -> RTMDKMemory:
    """Initialize or load RTMDK memory.

    Configuration is loaded from preset (RTMDK_PRESET env var, default "production")
    with individual field overrides via RTMDK_* env vars.
    """
    preset_name = os.getenv("RTMDK_PRESET", "production")
    preset_fn = getattr(RTMDKConfig, preset_name, None)
    if preset_fn is None:
        logger.warning(
            f"Unknown preset '{preset_name}', falling back to 'production'")
        preset_fn = RTMDKConfig.production

    # Preset creates the base config, env vars override individual fields
    from typing import Callable, cast
    config = cast(Callable[[], RTMDKConfig], preset_fn)()

    logger.info(f"Memory config preset: {preset_name}")
    logger.info(f"  latent_dim={config.latent_dim}, decay={config.decay_rate}")
    logger.info(f"  tension={config.tension_threshold}, top_k={config.top_k}")

    load_path = MEMORY_FILE
    if not os.path.exists(load_path):
        msgpack_path = os.path.splitext(load_path)[0] + ".msgpack"
        if os.path.exists(msgpack_path):
            load_path = msgpack_path

    if os.path.exists(load_path):
        try:
            mem = RTMDKMemory.import_field(
                load_path, get_embedding, wal_path=load_path + ".wal")
            if mem.field is not None:
                logger.info(
                    f"Loaded memory from {load_path}: {len(mem.field.nodes)} nodes")
            return mem
        except Exception:
            logger.warning(
                "Failed to load memory from %s",
                load_path,
                exc_info=True)
            import shutil

            backup_path = load_path + f".corrupted.{int(time.time())}"
            try:
                shutil.copy2(load_path, backup_path)
                os.remove(load_path)
            except Exception:
                pass

    save_path = _get_save_path(MEMORY_FILE)
    mem = RTMDKMemory(
        config=config,
        embedder=get_embedding,
        wal_path=save_path + ".wal")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mem.export_field(save_path)
        os.chmod(save_path, 0o600)  # Secure file permissions
        logger.info(f"Created new memory file at {save_path}")
    except Exception:
        logger.warning("Failed to create initial memory file", exc_info=True)

    return mem


def build_system_prompt(
        user_messages: List[ChatMessage],
        session_id: str) -> str:
    """Build system prompt with RTMDK context."""
    last_user = ""
    for msg in reversed(user_messages):
        if msg.role == "user":
            last_user = msg.content
            break

    ctx = {"rtmdk_context": ""}
    if last_user and memory:
        try:
            ctx = memory.load_memory_variables(
                {"input": last_user, "session_id": session_id})
        except Exception:
            logger.warning("Memory query failed", exc_info=True)

    # Check for custom system prompt (env var file)
    prompt_file = os.getenv("RTMDK_SYSTEM_PROMPT_FILE")

    base_prompt: Optional[str] = None
    # Priority: env var file > env var text > config.system_prompt > None
    if prompt_file and os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                base_prompt = f.read().strip()
        except Exception:
            logger.warning("Failed to read prompt file", exc_info=True)
            base_prompt = memory.config.system_prompt if memory else None
    else:
        env_prompt = os.getenv("RTMDK_SYSTEM_PROMPT")
        if env_prompt is not None:
            base_prompt = env_prompt if env_prompt else None
        elif memory:
            base_prompt = memory.config.system_prompt
        else:
            base_prompt = None

    system_prompt = base_prompt or ""
    if ctx["rtmdk_context"] and ctx["rtmdk_context"] not in (
            "No relevant memory.", "[]"):
        system_prompt += (
            f"\n\nRelevant memories:\n{ctx['rtmdk_context']}\n\n"
            "Use these memories to provide accurate, context-aware answers."
        )
    return system_prompt


def _get_save_path(base_path: str) -> str:
    """Select msgpack path if available, otherwise json."""
    try:
        pass

        if not base_path.endswith(".msgpack"):
            return os.path.splitext(base_path)[0] + ".msgpack"
    except ImportError:
        pass
    return base_path


def auto_save():
    """Auto-save memory to file if state changed since last save."""
    if not memory:
        return
    if not memory.field._dirty:
        logger.debug("Auto-save skipped: no changes since last save")
        return
    try:
        save_path = _get_save_path(MEMORY_FILE)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        memory.export_field(save_path)
        logger.debug(
            f"Auto-saved memory: {len(memory.field.nodes)} nodes to {save_path}")
    except Exception:
        logger.exception("Auto-save failed")


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================


# Production module instances (initialized at startup)
query_cache: Optional[QueryCache] = None
embedding_cache: Optional[EmbeddingCache] = None
context_optimizer: Optional[ContextOptimizer] = None
health_monitor: Optional[HealthMonitor] = None
analytics_dashboard: Optional[Any] = None
api_key_manager: Optional[APIKeyManager] = None
tenant_rate_limiter: Optional[TenantRateLimiter] = None
webhook_manager: Optional[WebhookManager] = None
audit_log: Optional[AuditLog] = None
retention_manager: Optional[RetentionManager] = None
redis_query_cache: Optional[RedisQueryCache] = None
redis_embedding_cache: Optional[RedisEmbeddingCache] = None
encryption_manager: Optional[EncryptionManager] = None
telemetry_manager: Optional[TelemetryManager] = None


# ============================================================================
# OPENAI-COMPATIBLE ENDPOINTS
# ============================================================================


@app.get("/v1/models")
async def list_models():
    """List available models."""
    if chat_model:
        return {"object": "list",
                "data": [{"id": chat_model,
                          "object": "model",
                          "created": int(time.time()),
                          "owned_by": "lm-studio"},
                         {"id": "rtmdk",
                          "object": "model",
                          "created": int(time.time()),
                          "owned_by": "rtmdk"},
                         ],
                }
    return {"object": "list",
            "data": [{"id": "rtmdk",
                      "object": "model",
                      "created": int(time.time()),
                      "owned_by": "rtmdk"}],
            }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Chat completions with RTMDK memory context."""
    if not lm_studio_available:
        raise HTTPException(status_code=503, detail="LM Studio not available")

    system_prompt = await run_sync(build_system_prompt, req.messages, req.session_id)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in req.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # Save user input to memory
    if memory and req.messages:
        last_user = next(
            (m.content for m in reversed(
                req.messages) if m.role == "user"), "")
        if last_user:
            try:
                await run_sync(memory.save_context, {"input": last_user, "session_id": req.session_id}, {"output": ""})
            except Exception:
                logger.warning("Memory save failed", exc_info=True)

    lm_timeout = int(os.getenv("RTMDK_LM_STUDIO_TIMEOUT", "120"))
    request_model = req.model if req.model and req.model != "rtmdk" else None
    actual_model = request_model or chat_model or "local-model"

    async def _fetch_chat():
        if _PROMETHEUS_AVAILABLE:
            _metric_lm_requests.labels(endpoint="chat").inc()
        return await http_client.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": actual_model,
                "messages": messages,
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
                "stream": req.stream,
            },
            timeout=lm_timeout,
            stream=req.stream,
        )

    resp = await llm_chat_circuit.call(_fetch_chat)
    if resp is None:
        if _PROMETHEUS_AVAILABLE:
            _metric_lm_errors.labels(endpoint="chat").inc()
        raise HTTPException(status_code=503,
                            detail="LM Studio unavailable (circuit open)")

    if req.stream:

        async def stream_generator():
            try:
                async for line in resp.aiter_lines():
                    if line:
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
            except Exception:
                logger.exception("Streaming error")
            finally:
                if memory:
                    try:
                        last_user = next(
                            (m.content for m in reversed(
                                req.messages) if m.role == "user"), "")
                        if last_user:
                            await run_sync(
                                memory.save_context,
                                {"input": last_user, "session_id": req.session_id},
                                {"output": "[streamed]"},
                            )
                    except Exception:
                        pass

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"},
        )

    data = resp.json()
    data["model"] = actual_model
    response_content = data["choices"][0]["message"]["content"]

    if memory and req.messages:
        try:
            last_user = next(
                (m.content for m in reversed(
                    req.messages) if m.role == "user"), "")
            if last_user:
                await run_sync(
                    memory.save_context,
                    {"input": last_user, "session_id": req.session_id},
                    {"output": response_content},
                )
        except Exception:
            logger.warning("Memory update failed", exc_info=True)

    return data


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    """Create embeddings."""
    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = []
    for i, text in enumerate(inputs):
        embedding = await get_embedding(text)
        data.append(
            {
                "object": "embedding",
                "embedding": embedding.tolist(),
                "index": i,
            }
        )
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {
            "prompt_tokens": sum(len(t.split()) for t in inputs),
            "total_tokens": sum(len(t.split()) for t in inputs),
        },
    }


# ============================================================================
# MEMORY QUERY ENDPOINTS
# ============================================================================


class MemoryQueryRequest(BaseModel):
    """Query the memory field."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results")
    threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the capital of France?",
                "top_k": 5,
                "threshold": 0.5,
            }
        }
    }


class MemoryQueryPipelineRequest(BaseModel):
    """Query the memory field using the explicit pipeline API."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results")
    session_id: Optional[str] = Field(None, description="Session ID for session boosting")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the capital of France?",
                "top_k": 5,
                "session_id": "sess_123",
            }
        }
    }


class BatchQueryRequest(BaseModel):
    """Batch query the memory field."""

    queries: List[str] = Field(..., min_length=1,
                               description="List of search queries")
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of results per query")
    threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "queries": ["capital of France", "largest ocean"],
                "top_k": 3,
                "threshold": 0.5,
            }
        }
    }


class AnalyticsTrackRequest(BaseModel):
    """Track a custom analytics event."""

    event_type: str = Field(..., min_length=1)
    properties: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "event_type": "user_query",
                "properties": {"topic": "science"},
                "session_id": "abc123",
            }
        }
    }


class CreateAPIKeyRequest(BaseModel):
    """Create a new API key for a tenant."""

    tenant_id: str = Field(..., min_length=1)
    name: str = Field(default="")
    rate_limit_override: Optional[Dict[str, int]] = Field(default=None)


class RevokeAPIKeyRequest(BaseModel):
    """Revoke an existing API key."""

    key_hash: str = Field(..., min_length=1)


class CreateNodeRequest(BaseModel):
    """Create a new memory node."""

    content: str = Field(..., min_length=1)
    node_id: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateNodeRequest(BaseModel):
    """Update an existing memory node."""

    content: Optional[str] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class BatchIngestRequest(BaseModel):
    """Batch ingest documents into memory."""

    documents: List[str] = Field(..., min_length=1, max_length=1000)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    node_ids: Optional[List[str]] = Field(default=None)


class MemoryImportRequest(BaseModel):
    """Import memory nodes from JSON payload."""

    nodes: List[Dict[str, Any]] = Field(..., min_length=1)
    clear_existing: bool = Field(default=False)


class WebhookSubscribeRequest(BaseModel):
    """Subscribe to webhook events."""

    url: str = Field(..., min_length=1)
    events: List[str] = Field(..., min_length=1)
    secret: Optional[str] = Field(default=None)


class WebhookUnsubscribeRequest(BaseModel):
    """Unsubscribe from webhook events."""

    subscription_id: str = Field(..., min_length=1)


class SOTBootstrapRequest(BaseModel):
    """Bootstrap SOT from a corpus of texts."""

    texts: List[str] = Field(..., min_length=1)
    teacher_model: Optional[str] = Field(default=None)


class ReplicationMutationRequest(BaseModel):
    """Receive a mutation from a peer node."""

    model_config = ConfigDict(extra="allow")


@app.post("/v1/memory/query")
async def memory_query(req: MemoryQueryRequest):
    """Query memory and return ranked results."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    _metric_queries.inc()
    t0 = time.time()
    try:
        # Query cache check
        if query_cache is not None:
            cached = query_cache.get(req.query)
            if cached is not None:
                _metric_query_dur.observe(time.time() - t0)
                return cached

        embedding = await _get_embedding_cached(req.query)
        assert memory.field is not None
        results = await run_sync(memory.field.query, embedding, top_k=req.top_k)
        # Filter by threshold and format
        formatted = []
        for nid, score, node in results:
            if score < req.threshold:
                continue
            formatted.append(
                {
                    "id": nid,
                    "content": (
                        node.content.get("content", node.content)
                        if isinstance(node.content, dict)
                        else str(node.content)
                    ),
                    "score": round(float(score), 4),
                }
            )
        resp = {
            "query": req.query,
            "results": formatted,
            "total": len(formatted),
        }
        if query_cache is not None:
            query_cache.put(req.query, resp)
        _metric_query_dur.observe(time.time() - t0)
        return resp
    except Exception as exc:
        _metric_query_dur.observe(time.time() - t0)
        logger.warning("Memory query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/memory/query_pipeline")
async def memory_query_pipeline(req: MemoryQueryPipelineRequest):
    """Query memory using the explicit pipeline API with per-stage metrics."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    t0 = time.time()
    try:
        result = await memory.retrieve_nodes_pipeline_async(
            req.query,
            top_k=req.top_k,
            session_id=req.session_id,
        )
        # Format results
        formatted = []
        for nid, score, node in result["results"]:
            formatted.append({
                "id": nid,
                "content": (
                    node.content.get("content", node.content)
                    if isinstance(node.content, dict)
                    else str(node.content)
                ),
                "score": round(float(score), 4),
            })

        resp = {
            "query": req.query,
            "results": formatted,
            "route": result.get("route"),
            "explanations": result.get("explanations", []),
            "metrics": result.get("metrics", {}),
            "total": len(formatted),
        }
        if pipeline_metrics_store is not None:
            pipeline_metrics_store.write(result.get("metrics", {}))
        _metric_query_dur.observe(time.time() - t0)
        return resp
    except Exception as exc:
        _metric_query_dur.observe(time.time() - t0)
        logger.warning("Pipeline query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/memory/pipeline/stream")
async def memory_pipeline_stream(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(5, ge=1, le=50, description="Number of results"),
    session_id: Optional[str] = Query(None, description="Session ID"),
):
    """Stream pipeline stage events via Server-Sent Events.

    Each event carries the completion status of a single stage,
    enabling live progress bars in dashboards and debug UIs.

    Example (JavaScript):
        const es = new EventSource(
            '/v1/memory/pipeline/stream?query=hello&top_k=5'
        );
        es.addEventListener('message', e => console.log(JSON.parse(e.data)));
    """
    if not memory or not memory.field:
        async def _error():
            yield f"data: {json.dumps({'event': 'error', 'message': 'Memory not initialized'})}\n\n"
        return StreamingResponse(_error(), media_type="text/event-stream")

    from rtmdk.pipeline.streaming import StreamingPipelineExecutor

    pipeline = memory.build_pipeline()
    streamer = StreamingPipelineExecutor(pipeline.stages)

    async def _generator():
        async for chunk in streamer.run_async(query, top_k=top_k, session_id=session_id):
            yield chunk

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/memory/pipeline/health")
async def memory_pipeline_health():
    """Return per-stage health status for the pipeline.

    Useful for load balancers and monitoring dashboards to determine
    whether the retrieval pipeline is healthy or degraded.
    """
    if not memory or not memory.field:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    pipeline = memory.build_pipeline()
    stages_health = []
    degraded_count = 0
    open_breakers = 0

    for stage in pipeline.stages:
        breaker_state = None
        if stage.circuit_breaker is not None:
            breaker_state = stage.circuit_breaker.state.value
            if breaker_state == "open":
                open_breakers += 1

        health = {
            "name": stage.name,
            "enabled": stage.enabled,
            "breaker_state": breaker_state,
            "has_fallback": hasattr(stage, "fallback") and stage.fallback is not None,
        }
        stages_health.append(health)

    overall = "healthy"
    if open_breakers > 0:
        overall = "degraded"
    if open_breakers >= len(stages_health) // 2:
        overall = "unhealthy"

    return {
        "overall": overall,
        "stages": stages_health,
        "open_breakers": open_breakers,
        "total_stages": len(stages_health),
    }


@app.get("/v1/memory/pipeline/prometheus")
async def memory_pipeline_prometheus():
    """Return pipeline metrics in Prometheus exposition format.

    Compatible with Prometheus scraping and Grafana dashboards.
    """
    if not memory or not memory.field:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    pipeline = memory.build_pipeline()
    lines = [
        "# HELP rtmdk_pipeline_stages_total Number of configured pipeline stages",
        "# TYPE rtmdk_pipeline_stages_total gauge",
        f"rtmdk_pipeline_stages_total {len(pipeline.stages)}",
        "",
        "# HELP rtmdk_pipeline_stage_enabled Whether a stage is enabled",
        "# TYPE rtmdk_pipeline_stage_enabled gauge",
    ]
    for stage in pipeline.stages:
        enabled = 1 if stage.enabled else 0
        lines.append(f'rtmdk_pipeline_stage_enabled{{stage="{stage.name}"}} {enabled}')

    lines.extend([
        "",
        "# HELP rtmdk_pipeline_breaker_state Circuit breaker state (0=closed, 1=half_open, 2=open)",
        "# TYPE rtmdk_pipeline_breaker_state gauge",
    ])
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    for stage in pipeline.stages:
        if stage.circuit_breaker is not None:
            state_val = state_map.get(stage.circuit_breaker.state.value, -1)
            lines.append(f'rtmdk_pipeline_breaker_state{{stage="{stage.name}"}} {state_val}')

    # Include metrics store stats if available
    if pipeline_metrics_store is not None:
        summary = pipeline_metrics_store.summary()
        lines.extend([
            "",
            "# HELP rtmdk_pipeline_queries_total Total pipeline queries",
            "# TYPE rtmdk_pipeline_queries_total counter",
            f"rtmdk_pipeline_queries_total {summary.get('queries', 0)}",
        ])
        for stage_name, stage_data in summary.get("stages", {}).items():
            lat = stage_data.get("latency_ms", {})
            if lat:
                lines.append(f'rtmdk_pipeline_stage_latency_ms{{stage="{stage_name}",quantile="0.5"}} {lat.get("median", 0)}')
                lines.append(f'rtmdk_pipeline_stage_latency_ms{{stage="{stage_name}",quantile="0.95"}} {lat.get("p95", 0)}')
            err_count = stage_data.get("errors", 0)
            lines.append(f'rtmdk_pipeline_stage_errors_total{{stage="{stage_name}"}} {err_count}')

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


@app.get("/v1/memory/pipeline/metrics")
async def memory_pipeline_metrics_summary(
    since: Optional[float] = Query(None, description="Unix timestamp — only metrics after this time"),
    stage: Optional[str] = Query(None, description="Filter to a single stage name"),
):
    """Return aggregated pipeline metrics summary.

    Query parameters:
        since — Unix timestamp for time-range filtering
        stage — filter metrics to a single stage (e.g. embed, retrieve)
    """
    if pipeline_metrics_store is None:
        return {"enabled": False, "message": "Pipeline metrics store not configured"}
    summary = pipeline_metrics_store.summary(since=since, stage_filter=stage)
    summary["enabled"] = True
    return summary


@app.post("/v1/memory/batch_query")
async def memory_batch_query(req: BatchQueryRequest):
    """Batch query memory for multiple queries — uses true batch resonance."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    _metric_queries.inc(len(req.queries))
    t0 = time.time()
    try:
        # Phase 1: per-query cache check and embedding gathering
        uncached_indices: List[int] = []
        uncached_queries: List[str] = []
        embeddings_list: List[Any] = []
        responses: List[Dict] = [{} for _ in req.queries]

        for i, q in enumerate(req.queries):
            if query_cache is not None:
                cached = query_cache.get(q)
                if cached is not None:
                    responses[i] = {
                        "query": q, "results": cached.get("results", []), "cached": True}
                    continue
            uncached_indices.append(i)
            uncached_queries.append(q)
            emb = await _get_embedding_cached(q)
            embeddings_list.append(emb)

        # Phase 2: batch resonance for uncached queries
        if embeddings_list:
            batch_results = await run_sync(
                memory.batch_query,
                embeddings_list,
                top_k=req.top_k)
            for offset, i in enumerate(uncached_indices):
                q = req.queries[i]
                results = batch_results[offset]
                formatted = []
                for nid, score, node in results:
                    if score < req.threshold:
                        continue
                    formatted.append(
                        {
                            "id": nid,
                            "content": (
                                node.content.get("content", node.content)
                                if isinstance(node.content, dict)
                                else str(node.content)
                            ),
                            "score": round(float(score), 4),
                        }
                    )
                resp = {"query": q, "results": formatted}
                if query_cache is not None:
                    query_cache.put(
                        q, {"query": q, "results": formatted, "total": len(formatted)})
                responses[i] = resp

        _metric_query_dur.observe(time.time() - t0)
        return {
            "queries": len(req.queries),
            "results": responses,
            "latency_ms": round((time.time() - t0) * 1000, 2)}
    except Exception as exc:
        _metric_query_dur.observe(time.time() - t0)
        logger.warning("Batch memory query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# MEMORY NODE CRUD ENDPOINTS
# ============================================================================


@app.post("/v1/memory/nodes")
async def create_node(req: CreateNodeRequest):
    """Create a new memory node."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    assert memory.field is not None
    try:
        embedding = await _get_embedding_cached(req.content)
        content_dict = {"content": req.content, **req.metadata}
        nid = memory.field.add_node(
            embedding=embedding,
            content=content_dict,
            node_id=req.node_id,
        )
        if audit_log is not None:
            audit_log.record(
                action="create_node",
                resource=nid,
                details={"content_preview": req.content[:100]},
            )
        return {"id": nid, "status": "created"}
    except Exception as exc:
        logger.warning("Create node failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/memory/nodes/{node_id}")
async def get_node(node_id: str):
    """Get a memory node by ID."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    assert memory.field is not None
    node = memory.field.nodes.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return {
        "id": node_id,
        "content": node.content,
        "salience": node.salience,
        "created_at": getattr(node, "created_at", None),
        "last_accessed": getattr(node, "last_accessed", None),
    }


@app.put("/v1/memory/nodes/{node_id}")
async def update_node(node_id: str, req: UpdateNodeRequest):
    """Update an existing memory node."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    assert memory.field is not None
    node = memory.field.nodes.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    try:
        if req.content is not None:
            embedding = await _get_embedding_cached(req.content)
            node.latent_pos = embedding
            if isinstance(node.content, dict):
                node.content["content"] = req.content
            else:
                node.content = req.content
        if req.metadata is not None:
            if isinstance(node.content, dict):
                node.content.update(req.metadata)
        if audit_log is not None:
            audit_log.record(
                action="update_node",
                resource=node_id,
            )
        return {"id": node_id, "status": "updated"}
    except Exception as exc:
        logger.warning("Update node failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/v1/memory/nodes/{node_id}")
async def delete_node(node_id: str):
    """Delete a memory node by ID."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    assert memory.field is not None
    node = memory.field.nodes.get(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    try:
        memory.field.delete_nodes([node_id])
        if audit_log is not None:
            audit_log.record(
                action="delete_node",
                resource=node_id,
            )
        return {"id": node_id, "status": "deleted"}
    except Exception as exc:
        logger.warning("Delete node failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/memory/nodes")
async def list_nodes(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List memory nodes with pagination."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    assert memory.field is not None
    nodes = list(memory.field.nodes.items())
    total = len(nodes)
    page = nodes[offset:offset + limit]
    results = []
    for nid, node in page:
        results.append({
            "id": nid,
            "content": node.content if isinstance(node.content, dict) else {"content": str(node.content)},
            "salience": node.salience,
        })
    return {"total": total, "offset": offset, "limit": limit, "nodes": results}


@app.post("/v1/memory/batch_ingest")
async def batch_ingest(req: BatchIngestRequest):
    """Batch ingest documents into memory."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    assert memory.field is not None
    try:
        t0 = time.time()
        created = []
        for idx, doc in enumerate(req.documents):
            embedding = await _get_embedding_cached(doc)
            content_dict = {"content": doc, **(req.metadata or {})}
            nid = req.node_ids[idx] if req.node_ids and idx < len(req.node_ids) else None
            node_id = memory.field.add_node(
                embedding=embedding,
                content=content_dict,
                node_id=nid,
            )
            created.append(node_id)
        latency_ms = round((time.time() - t0) * 1000, 2)
        return {
            "ingested": len(created),
            "node_ids": created,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        logger.warning("Batch ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/memory/export")
async def memory_export():
    """Export all memory nodes as JSON."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    assert memory.field is not None
    try:
        nodes = []
        for nid, node in memory.field.nodes.items():
            nodes.append({
                "id": nid,
                "content": node.content,
                "latent_pos": (
                    node.latent_pos.tolist()
                    if hasattr(node.latent_pos, "tolist")
                    else list(node.latent_pos)
                ),
                "salience": node.salience,
                "created_at": getattr(node, "created_at", None),
                "last_accessed": getattr(node, "last_accessed", None),
            })
        return {"nodes": nodes, "total": len(nodes), "exported_at": time.time()}
    except Exception as exc:
        logger.warning("Memory export failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/memory/import")
async def memory_import(req: MemoryImportRequest):
    """Import memory nodes from JSON payload."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    assert memory.field is not None
    try:
        import numpy as np
        t0 = time.time()
        if req.clear_existing:
            memory.field.nodes.clear()
        created = []
        for node_data in req.nodes:
            nid = node_data.get("id")
            emb = np.array(node_data.get("latent_pos", node_data.get("embedding", [])), dtype=np.float32)
            content = node_data.get("content", {})
            node_id = memory.field.add_node(
                embedding=emb,
                content=content,
                node_id=nid,
            )
            created.append(node_id)
        latency_ms = round((time.time() - t0) * 1000, 2)
        return {
            "imported": len(created),
            "node_ids": created,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        logger.warning("Memory import failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    """Health check with production metrics."""
    base = {
        "status": "ok",
        "version": "8.2.0",
        "lm_studio": lm_studio_available,
        "memory_nodes": len(memory.field.nodes) if memory else 0,
    }
    if health_monitor is not None:
        try:
            h = health_monitor.check_health()
            base["health"] = h
        except Exception:
            logger.exception("Health monitor check failed")
    if query_cache is not None:
        base["query_cache"] = {
            "hit_rate": round(query_cache.hit_rate, 3),
            "size": len(query_cache._cache),
        }
    if embedding_cache is not None:
        base["embedding_cache"] = {
            "hit_rate": round(embedding_cache.hit_rate, 3),
            "memory_size": len(embedding_cache.memory_cache),
        }
    return base


@app.get("/health/deep")
async def health_deep():
    """Deep health check with integrity probes."""
    checks = {}
    overall = "ok"

    # Memory field check
    if memory and memory.field is not None:
        field = memory.field
        checks["memory_field"] = {
            "nodes": len(field.nodes),
            "status": "ok",
        }
        # HNSW integrity check
        hnsw = getattr(field, "hnsw_index", None)
        if hnsw is not None:
            try:
                hnsw_size = getattr(hnsw, "get_current_count", lambda: -1)()
                checks["hnsw"] = {"status": "ok", "indexed_nodes": hnsw_size}
            except Exception as exc:
                checks["hnsw"] = {"status": "error", "error": str(exc)}
                overall = "degraded"
        # Embedding dimension consistency
        try:
            expected_dim = memory.config.latent_dim
            sample_nodes = list(field.nodes.values())[:5]
            dim_ok = all(
                getattr(n, "latent_pos", None) is not None and len(n.latent_pos) == expected_dim
                for n in sample_nodes
            )
            checks["embedding_dims"] = {
                "status": "ok" if dim_ok else "error",
                "expected": expected_dim,
            }
            if not dim_ok:
                overall = "degraded"
        except Exception as exc:
            checks["embedding_dims"] = {"status": "error", "error": str(exc)}
            overall = "degraded"
    else:
        checks["memory_field"] = {"status": "error", "error": "Memory not initialized"}
        overall = "error"

    # WAL backlog check
    wal = getattr(memory.field if memory else None, "wal", None)
    if wal is not None:
        try:
            backlog = len(wal._buffer) if hasattr(wal, "_buffer") else 0
            checks["wal"] = {"status": "ok", "backlog": backlog}
        except Exception as exc:
            checks["wal"] = {"status": "error", "error": str(exc)}
    else:
        checks["wal"] = {"status": "ok", "backlog": 0}

    # Async index builder check
    aib = getattr(memory.field if memory else None, "async_index_builder", None)
    if aib is not None:
        try:
            pending = len(aib._pending) if hasattr(aib, "_pending") else 0
            checks["async_index"] = {"status": "ok", "pending": pending}
        except Exception as exc:
            checks["async_index"] = {"status": "error", "error": str(exc)}
    else:
        checks["async_index"] = {"status": "ok", "pending": 0}

    # Active requests check
    checks["active_requests"] = {"count": _active_requests}

    return {
        "status": overall,
        "version": "8.2.0",
        "checks": checks,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501,
                            detail="prometheus-client not installed")
    if memory:
        node_count = len(memory.field.nodes)
        _metric_nodes.set(node_count)
        try:
            from rtmdk.production.metrics import update_node_count
            update_node_count(node_count)
        except Exception:
            pass
        sot = getattr(memory.field, "sot_tokenizer", None)
        if sot:
            mode = getattr(sot, "tokenization_mode", "byte")
            vocab_size = len(getattr(sot, "word_to_id", {}))
            _metric_sot_vocab.labels(mode=mode).set(vocab_size)
            coocc = getattr(sot, "cooccurrence_store", None)
            if coocc is not None:
                _metric_sot_cooccurrence.set(len(coocc))
            else:
                _metric_sot_cooccurrence.set(0)
            boot_time = getattr(sot, "_bootstrap_time", None)
            if boot_time is not None:
                _metric_sot_bootstrap_time.set(boot_time)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# ANALYTICS DASHBOARD ENDPOINTS
# ============================================================================


@app.get("/v1/analytics/overview")
async def analytics_overview():
    """Dashboard overview with key metrics."""
    if analytics_dashboard is None:
        raise HTTPException(status_code=503,
                            detail="Analytics dashboard not available")
    try:
        return analytics_dashboard.get_overview()
    except Exception as exc:
        logger.warning("Analytics overview failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/analytics/memory")
async def analytics_memory():
    """Memory-specific analytics."""
    if analytics_dashboard is None:
        raise HTTPException(status_code=503,
                            detail="Analytics dashboard not available")
    try:
        return analytics_dashboard.get_memory_analytics()
    except Exception as exc:
        logger.warning("Analytics memory failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/analytics/events")
async def analytics_events(limit: int = Query(50, ge=1, le=500),
                           event_type: Optional[str] = Query(None)):
    """Recent event log with optional filtering."""
    if analytics_dashboard is None:
        raise HTTPException(status_code=503,
                            detail="Analytics dashboard not available")
    try:
        return analytics_dashboard.get_event_series(
            limit=limit,
            event_type=event_type,
        )
    except Exception as exc:
        logger.warning("Analytics events failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/analytics/report")
async def analytics_report():
    """Full analytics report combining all metrics."""
    if analytics_dashboard is None:
        raise HTTPException(status_code=503,
                            detail="Analytics dashboard not available")
    try:
        return analytics_dashboard.get_report()
    except Exception as exc:
        logger.warning("Analytics report failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/analytics/track")
async def analytics_track(req: AnalyticsTrackRequest):
    """Track a custom analytics event."""
    if analytics_dashboard is None:
        raise HTTPException(status_code=503,
                            detail="Analytics dashboard not available")
    try:
        analytics_dashboard.track_event(
            event_type=req.event_type,
            properties=req.properties,
            session_id=req.session_id,
        )
        return {"status": "ok"}
    except Exception as exc:
        logger.warning("Analytics track failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# AUDIT LOG ENDPOINTS
# ============================================================================


@app.get("/v1/admin/audit-log")
async def audit_log_query(
    request: Request,
    actor: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    since: Optional[float] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Query audit log entries (admin only)."""
    _require_admin(request)
    if audit_log is None:
        raise HTTPException(status_code=503, detail="Audit log not available")
    try:
        entries = audit_log.query(actor=actor, action=action, since=since, limit=limit)
        return {"entries": entries, "count": len(entries)}
    except Exception as exc:
        logger.warning("Audit log query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/admin/retention")
async def retention_stats(request: Request):
    """Get retention manager statistics (admin only)."""
    _require_admin(request)
    if retention_manager is None:
        raise HTTPException(status_code=503, detail="Retention manager not available")
    return retention_manager.stats()


@app.get("/v1/admin/cache")
async def cache_stats(request: Request):
    """Get cache statistics (admin only)."""
    _require_admin(request)
    return {
        "redis_query": redis_query_cache.stats() if redis_query_cache else None,
        "redis_embedding": redis_embedding_cache.stats() if redis_embedding_cache else None,
    }


@app.get("/v1/admin/encryption")
async def encryption_status(request: Request):
    """Get encryption status (admin only)."""
    _require_admin(request)
    if encryption_manager is None:
        raise HTTPException(status_code=503, detail="Encryption manager not available")
    return {"enabled": encryption_manager.enabled}


@app.get("/v1/admin/telemetry")
async def telemetry_status(request: Request):
    """Get telemetry status (admin only)."""
    _require_admin(request)
    if telemetry_manager is None:
        raise HTTPException(status_code=503, detail="Telemetry manager not available")
    return {"enabled": telemetry_manager.enabled}


# ============================================================================
# WEBSOCKET STREAMING
# ============================================================================


@app.websocket("/ws/memory")
async def memory_websocket(websocket: WebSocket):
    """Real-time WebSocket for memory events."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                action = msg.get("action")
                if action == "query":
                    query = msg.get("query", "")
                    top_k = msg.get("top_k", 5)
                    if memory and memory.field:
                        embedding = await _get_embedding_cached(query)
                        results = await run_sync(memory.field.query, embedding, top_k=top_k)
                        out = []
                        for nid, score, node in results:
                            content = ""
                            if hasattr(node, "content"):
                                if isinstance(node.content, dict):
                                    content = node.content.get("text", str(node.content))
                                else:
                                    content = str(node.content)
                            out.append({
                                "node_id": nid,
                                "score": score,
                                "content": content,
                            })
                        await websocket.send_json({"type": "query_results", "results": out})
                    else:
                        await websocket.send_json({"type": "error", "message": "Memory not ready"})
                elif action == "query_pipeline":
                    query = msg.get("query", "")
                    top_k = msg.get("top_k", 5)
                    session_id = msg.get("session_id")
                    use_stream = msg.get("stream", False)
                    if memory and memory.field:
                        try:
                            if use_stream:
                                from rtmdk.pipeline.streaming import StreamingPipelineExecutor
                                pipeline = memory.build_pipeline()
                                streamer = StreamingPipelineExecutor(pipeline.stages)
                                async for chunk in streamer.run_async(query, top_k=top_k, session_id=session_id):
                                    # chunk is already SSE-formatted "data: {...}\n\n"
                                    # strip prefix for WebSocket JSON
                                    raw = chunk.strip()
                                    if raw.startswith("data: "):
                                        raw = raw[6:]
                                    event_data = json.loads(raw)
                                    await websocket.send_json({
                                        "type": "pipeline_event",
                                        "event": event_data,
                                    })
                            else:
                                result = await memory.retrieve_nodes_pipeline_async(
                                    query, top_k=top_k, session_id=session_id
                                )
                                formatted = []
                                for nid, score, node in result["results"]:
                                    content = ""
                                    if hasattr(node, "content"):
                                        if isinstance(node.content, dict):
                                            content = node.content.get("text", str(node.content))
                                        else:
                                            content = str(node.content)
                                    formatted.append({
                                        "node_id": nid,
                                        "score": round(float(score), 4),
                                        "content": content,
                                    })
                                await websocket.send_json({
                                    "type": "pipeline_results",
                                    "query": query,
                                    "results": formatted,
                                    "route": result.get("route"),
                                    "metrics": result.get("metrics"),
                                    "total": len(formatted),
                                })
                        except Exception as exc:
                            await websocket.send_json({"type": "error", "message": str(exc)})
                    else:
                        await websocket.send_json({"type": "error", "message": "Memory not ready"})
                elif action == "ping":
                    await websocket.send_json({"type": "pong"})
                else:
                    await websocket.send_json({"type": "error", "message": f"Unknown action: {action}"})
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ============================================================================
# SOT ENDPOINTS
# ============================================================================


@app.get("/v1/sot/status")
async def sot_status():
    """Get SOT (Self-Organizing Tokenizer) status."""
    if memory is None or memory.field is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    sot = memory.field.sot_tokenizer
    if sot is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "vocab_size": len(sot.token_embeddings),
        "max_vocab": sot.max_vocab,
        "tokenization_mode": getattr(sot, "tokenization_mode", "unknown"),
        "cooccurrence_size": len(sot.cooccurrence) if hasattr(sot, "cooccurrence") else 0,
    }


@app.get("/v1/sot/vocab")
async def sot_vocab(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
):
    """Inspect SOT vocabulary."""
    if memory is None or memory.field is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    sot = memory.field.sot_tokenizer
    if sot is None:
        raise HTTPException(status_code=503, detail="SOT not enabled")
    items = []
    word_map = getattr(sot, "word_to_id", {})
    for word, tid in list(word_map.items())[offset:offset + limit]:
        if search and search.lower() not in word.lower():
            continue
        items.append({"word": word, "token_id": tid})
    return {"items": items, "total": len(word_map), "limit": limit, "offset": offset}


@app.post("/v1/sot/bootstrap")
async def sot_bootstrap(req: SOTBootstrapRequest):
    """Bootstrap SOT from a corpus of texts."""
    if memory is None or memory.field is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    sot = memory.field.sot_tokenizer
    if sot is None:
        raise HTTPException(status_code=503, detail="SOT not enabled")
    texts = req.texts
    teacher = req.teacher_model
    if not texts:
        raise HTTPException(status_code=400, detail="texts required")

    async def _do_bootstrap():
        if teacher:
            memory.field.sot_bootstrap(texts, teacher_model=teacher)
        else:
            sot.warm_start_from_corpus(texts)

    try:
        await _sot_bootstrap_breaker.call(_do_bootstrap)
        return {"status": "bootstrapped", "vocab_size": len(sot.token_embeddings)}
    except Exception as exc:
        logger.exception("SOT bootstrap failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# REPLICATION ENDPOINTS (v8.2.1)
# ============================================================================


@app.post("/v1/replication/mutation")
async def replication_receive_mutation(req: ReplicationMutationRequest):
    """Receive a mutation from a peer node."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    rm = getattr(memory, "replication_manager", None)
    if rm is None or not rm.enabled:
        raise HTTPException(status_code=503, detail="Replication not enabled")
    payload = req.model_dump()
    clock = payload.get("_rep_clock", 0)
    origin = payload.get("_rep_origin", "unknown")
    rm._wal.append(clock, origin, payload)
    return {"status": "accepted", "clock": clock}


@app.get("/v1/replication/wal")
async def replication_get_wal(since: int = 0):
    """Return local WAL entries with clock > since."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    rm = getattr(memory, "replication_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="Replication not enabled")
    entries = rm.get_wal(since=since)
    return {"mutations": entries, "node_id": rm.node_id}


# ============================================================================
# API KEY MANAGEMENT ENDPOINTS
# ============================================================================


def _require_admin(request: Request):
    """Ensure request carries the legacy admin key."""
    if not ENABLE_API_AUTH:
        return
    api_key = getattr(request.state, "api_key", "")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")


@app.post("/v1/admin/api-keys")
async def create_api_key(req: CreateAPIKeyRequest, request: Request):
    """Create a new API key for a tenant (admin only)."""
    _require_admin(request)
    if api_key_manager is None:
        raise HTTPException(status_code=503, detail="API key manager not available")
    try:
        raw_key, record = api_key_manager.create_key(
            tenant_id=req.tenant_id,
            name=req.name,
            rate_limit_override=req.rate_limit_override,
        )
        return {
            "api_key": raw_key,
            "key_hash": record.key_hash,
            "tenant_id": record.tenant_id,
            "created_at": record.created_at,
        }
    except Exception as exc:
        logger.warning("Create API key failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/admin/api-keys")
async def list_api_keys(tenant_id: Optional[str] = Query(None), request: Request = ...):  # type: ignore[assignment]
    # FastAPI injects Request automatically
    """List API keys metadata (admin only)."""
    _require_admin(request)
    if api_key_manager is None:
        raise HTTPException(status_code=503, detail="API key manager not available")
    try:
        keys = api_key_manager.list_keys(tenant_id=tenant_id, include_revoked=True)
        return {"keys": keys}
    except Exception as exc:
        logger.warning("List API keys failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/admin/api-keys/revoke")
async def revoke_api_key(req: RevokeAPIKeyRequest, request: Request):
    """Revoke an API key by hash (admin only)."""
    _require_admin(request)
    if api_key_manager is None:
        raise HTTPException(status_code=503, detail="API key manager not available")
    try:
        ok = api_key_manager.revoke_key(req.key_hash)
        if not ok:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"status": "revoked", "key_hash": req.key_hash}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Revoke API key failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/v1/admin/api-keys/{key_hash}")
async def delete_api_key(key_hash: str, request: Request):
    """Permanently delete an API key (admin only)."""
    _require_admin(request)
    if api_key_manager is None:
        raise HTTPException(status_code=503, detail="API key manager not available")
    try:
        ok = api_key_manager.delete_key(key_hash)
        if not ok:
            raise HTTPException(status_code=404, detail="Key not found")
        return {"status": "deleted", "key_hash": key_hash}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Delete API key failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/v1/admin/tenants")
async def list_tenants(request: Request):
    """List all tenants with key counts (admin only)."""
    _require_admin(request)
    if api_key_manager is None:
        raise HTTPException(status_code=503, detail="API key manager not available")
    try:
        keys = api_key_manager.list_keys(include_revoked=True)
        tenants: Dict[str, Dict] = {}
        for k in keys:
            tid = k["tenant_id"]
            if tid not in tenants:
                tenants[tid] = {"tenant_id": tid, "total_keys": 0, "active_keys": 0}
            tenants[tid]["total_keys"] += 1
            if not k.get("revoked"):
                tenants[tid]["active_keys"] += 1
        return {"tenants": list(tenants.values())}
    except Exception as exc:
        logger.warning("List tenants failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/admin/config")
async def admin_config_reload(req: dict, request: Request):
    """Hot-reload configurable parameters (admin only).

    Body: {"decay_rate": 0.998, "top_k": 10}
    Only a whitelist of fields can be changed at runtime.
    """
    _require_admin(request)
    if memory is None or memory.field is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    whitelist = {
        "decay_rate", "top_k", "min_response", "bandwidth",
        "phase_coupling", "tension_threshold", "adaptive_threshold",
    }
    updated = []
    for key, value in req.items():
        if key not in whitelist:
            raise HTTPException(status_code=400, detail=f"Field '{key}' not allowed for hot reload")
        try:
            setattr(memory.config, key, value)
            updated.append(key)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid value for {key}: {exc}")
    return {"status": "updated", "fields": updated}


# ============================================================================
# WEBHOOK ENDPOINTS
# ============================================================================


@app.post("/v1/webhooks")
async def webhook_subscribe(req: WebhookSubscribeRequest, request: Request):
    """Subscribe to webhook events."""
    if webhook_manager is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")
    tenant_id = getattr(request.state, "tenant_id", None)
    try:
        sub = webhook_manager.subscribe(
            url=req.url,
            events=req.events,
            secret=req.secret,
            tenant_id=tenant_id,
        )
        return {
            "subscription_id": sub.id,
            "url": sub.url,
            "events": sub.events,
            "created_at": sub.created_at,
        }
    except Exception as exc:
        logger.warning("Webhook subscribe failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/v1/webhooks/{subscription_id}")
async def webhook_unsubscribe(subscription_id: str):
    """Unsubscribe from webhook events."""
    if webhook_manager is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")
    ok = webhook_manager.unsubscribe(subscription_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return {"status": "unsubscribed", "subscription_id": subscription_id}


@app.get("/v1/webhooks")
async def webhook_list(request: Request):
    """List active webhook subscriptions."""
    if webhook_manager is None:
        raise HTTPException(status_code=503, detail="Webhook manager not available")
    tenant_id = getattr(request.state, "tenant_id", None)
    subs = webhook_manager.list_subscriptions(tenant_id=tenant_id)
    return {
        "subscriptions": [
            {
                "id": s.id,
                "url": s.url,
                "events": s.events,
                "active": s.active,
                "tenant_id": s.tenant_id,
            }
            for s in subs
        ]
    }


# ============================================================================
# UX ENDPOINTS (from package)
# ============================================================================


def create_ux_router(memory_fn, config: Dict) -> "APIRouter":
    """Import and return UX router from the package."""
    from rtmdk_server_ux import create_ux_router as _create_ux

    return _create_ux(memory_fn, config)


def create_dashboard_router(memory_fn, config: Dict) -> "APIRouter":
    """Import and return Dashboard router from the package."""
    from rtmdk_dashboard_ui import create_dashboard_router as _create_dash

    return _create_dash(memory_fn, config)


_ux_config = {
    "RTMDK_BACKUP_DIR": os.path.join(
        os.path.expanduser("~"),
        ".rtmdk",
        "backups"),
    "RTMDK_SESSION_DIR": os.path.join(
        os.path.expanduser("~"),
        ".rtmdk",
        "sessions"),
    "RTMDK_CACHE_DIR": os.path.join(
        os.path.expanduser("~"),
        ".rtmdk",
        "embedding_cache"),
    "RTMDK_CACHE_MAX_SIZE": "100000",
}

app.include_router(create_ux_router(lambda: memory, _ux_config))
app.include_router(create_dashboard_router(lambda: memory, _ux_config))


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 60)
    print("  RTMDK Production API v8.2.0")
    print("  (No SillyTavern modules)")
    print("=" * 60)
    print()
    print(f"  Server: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"  Memory: {MEMORY_FILE}")
    print(f"  LM Studio: {LM_STUDIO_URL}")
    print(f"  API Key: {API_KEY}")
    print()
    print("  Endpoints:")
    print("    POST /v1/chat/completions    — Chat with memory")
    print("    POST /v1/embeddings          — Embeddings")
    print("    POST /v1/memory/query        — Query memory")
    print("    POST /v1/memory/batch_query  — Batch query memory")
    print("    GET  /v1/models              — List models")
    print("    GET  /health                 — Health check")
    print("    GET  /metrics                — Prometheus metrics")
    print("    GET  /dashboard              — Web UI Dashboard")
    print("    GET  /api/models             — UX model selector")
    print("    POST /api/config             — Runtime configuration")
    print("    GET  /v1/analytics/overview  — Dashboard overview")
    print("    GET  /v1/analytics/memory    — Memory analytics")
    print("    GET  /v1/analytics/events    — Event log")
    print("    GET  /v1/analytics/report    — Full analytics report")
    print()
    print("-" * 60)

    import uvicorn

    uvicorn.run(
        "rtmdk.server.app:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
