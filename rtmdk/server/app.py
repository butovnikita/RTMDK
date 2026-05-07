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
from typing import Dict, List, Optional

import httpx
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

# RTMDK package imports
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
from rtmdk.production.rate_limiter import RateLimiter
from rtmdk.production.query_cache import QueryCache
from rtmdk.production.embedding_cache import EmbeddingCache
from rtmdk.production.context_optimizer import ContextOptimizer
from rtmdk.production.health_monitor import HealthMonitor
from rtmdk.support.circuit_breaker import AsyncCircuitBreaker

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
    logger.info("Starting RTMDK Production API v8.0.0")
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
    asyncio.create_task(_health_check_loop())
    asyncio.create_task(_auto_save_loop())

    logger.info(f"Server ready on {SERVER_HOST}:{SERVER_PORT}")
    if memory.field is not None:
        logger.info(f"Memory nodes: {len(memory.field.nodes)}")
    logger.info(
        "Production modules: QueryCache, EmbeddingCache, ContextOptimizer, HealthMonitor enabled")

    yield

    logger.info("RTMDK server shutting down...")
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
    version="8.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enforce API key authentication and payload limits."""
    skip_auth_paths = [
        "/health",
        "/v1/models",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/dashboard"]
    if request.url.path in skip_auth_paths or request.url.path.startswith(
            "/api/"):
        return await call_next(request)

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
        return JSONResponse(
            status_code=413, content={
                "error": "Payload too large"})

    if ENABLE_API_AUTH:
        auth_header = request.headers.get("authorization", "")
        api_key = auth_header.replace(
            "Bearer ", "").replace(
            "bearer ", "") if auth_header else ""
        if not api_key:
            api_key = request.headers.get("x-api-key", "")
        if not api_key or api_key != API_KEY:
            return JSONResponse(
                status_code=401, content={
                    "error": "Unauthorized. Provide valid API key."})

    return await call_next(request)


# Phase 4: Rate limiter middleware
rate_limiter = RateLimiter(
    max_per_minute=int(os.getenv("RTMDK_RATE_LIMIT_PER_MINUTE", "60")),
    max_per_hour=int(os.getenv("RTMDK_RATE_LIMIT_PER_HOUR", "1000")),
    max_per_day=int(os.getenv("RTMDK_RATE_LIMIT_PER_DAY", "10000")),
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-client rate limits."""
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
    client_id = request.headers.get(
        "x-api-key", request.client.host if request.client else "unknown")
    if not rate_limiter.allow_request(client_id):
        remaining = rate_limiter.get_remaining(client_id)
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "remaining": remaining})
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
    """Get embedding from LM Studio or cache."""
    cached = embedder_cache.get(text)
    if cached is not None:
        return cached

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


@app.post("/v1/memory/batch_query")
async def memory_batch_query(req: BatchQueryRequest):
    """Batch query memory for multiple queries."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")

    _metric_queries.inc(len(req.queries))
    t0 = time.time()
    try:
        responses = []
        for q in req.queries:
            # Query cache check per query
            if query_cache is not None:
                cached = query_cache.get(q)
                if cached is not None:
                    responses.append(
                        {"query": q, "results": cached.get("results", []), "cached": True})
                    continue

            embedding = await _get_embedding_cached(q)
            assert memory.field is not None
            results = await run_sync(memory.field.query, embedding, top_k=req.top_k)
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
            responses.append(resp)
        _metric_query_dur.observe(time.time() - t0)
        return {
            "queries": len(
                req.queries),
            "results": responses,
            "latency_ms": round(
                (time.time() - t0) * 1000,
                2)}
    except Exception as exc:
        _metric_query_dur.observe(time.time() - t0)
        logger.warning("Batch memory query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    """Health check with production metrics."""
    base = {
        "status": "ok",
        "version": "8.0.0",
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


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501,
                            detail="prometheus-client not installed")
    if memory:
        _metric_nodes.set(len(memory.field.nodes))
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
    print("  RTMDK Production API v8.0.0")
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
