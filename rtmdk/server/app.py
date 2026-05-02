"""RTMDK Production Server — OpenAI-compatible API with Resonance-Topological Memory.

This is the CLEAN production version WITHOUT SillyTavern modules.
For development with SillyTavern support, use rtmdk_server.py instead.

Usage:
    python -m rtmdk
    python start_production.py
"""

import os
import sys
import time
import json
import asyncio
import logging
import logging.handlers
import httpx
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel

# RTMDK package imports
from rtmdk.memory.core import RTMDKMemory, RTMDKConfig
from rtmdk.production.rate_limiter import RateLimiter
from rtmdk.support.circuit_breaker import AsyncCircuitBreaker

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False

if _PROMETHEUS_AVAILABLE:
    _metric_nodes = Gauge("rtmdk_nodes_total", "Number of memory nodes")
    _metric_queries = Counter("rtmdk_queries_total", "Total memory queries")
    _metric_query_dur = Histogram("rtmdk_query_duration_seconds", "Query duration")
    _metric_consolidations = Counter("rtmdk_consolidations_total", "Total consolidations")
    _metric_security = Counter("rtmdk_security_violations_total", "Security violations")
    _metric_lm_requests = Counter("rtmdk_lm_requests_total", "LM Studio requests", ["endpoint"])
    _metric_lm_errors = Counter("rtmdk_lm_errors_total", "LM Studio errors", ["endpoint"])


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
MEMORY_FILE = os.getenv("RTMDK_MEMORY_FILE", os.path.join(os.path.expanduser("~"), ".rtmdk", "memory.json"))
EMBED_MODEL = os.getenv("RTMDK_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
API_KEY = os.getenv("RTMDK_API_KEY", "rtmdk-local")
ENABLE_LM_STUDIO = os.getenv("RTMDK_ENABLE_LM_STUDIO", "true").lower() == "true"
ENABLE_API_AUTH = os.getenv("RTMDK_ENABLE_API_AUTH", "true").lower() == "true"
MAX_PAYLOAD_SIZE = int(os.getenv("RTMDK_MAX_PAYLOAD_SIZE", "1048576"))
ALLOWED_ORIGINS = os.getenv("RTMDK_ALLOWED_ORIGINS", "*").split(",")

# Shared async HTTP client
http_client = httpx.AsyncClient()

# ============================================================================
# SIGNAL / LIFECYCLE HANDLERS
# ============================================================================

import atexit
import signal

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
    _log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.basicConfig(level=log_level, handlers=[_log_handler])
logger = logging.getLogger("rtmdk_production")

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="RTMDK Production API",
    description="OpenAI-compatible API with Resonance-Topological Memory (No SillyTavern)",
    version="8.0.0",
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
    skip_auth_paths = ["/health", "/v1/models", "/docs", "/openapi.json", "/redoc", "/dashboard"]
    if request.url.path in skip_auth_paths or request.url.path.startswith("/api/"):
        return await call_next(request)

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
        return JSONResponse(status_code=413, content={"error": "Payload too large"})

    if ENABLE_API_AUTH:
        auth_header = request.headers.get("authorization", "")
        api_key = auth_header.replace("Bearer ", "").replace("bearer ", "") if auth_header else ""
        if not api_key:
            api_key = request.headers.get("x-api-key", "")
        if not api_key or api_key != API_KEY:
            return JSONResponse(status_code=401, content={"error": "Unauthorized. Provide valid API key."})

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
    skip_paths = {"/health", "/v1/models", "/docs", "/openapi.json", "/redoc", "/dashboard", "/metrics"}
    if request.url.path in skip_paths or request.url.path.startswith("/api/"):
        return await call_next(request)
    client_id = request.headers.get("x-api-key", request.client.host if request.client else "unknown")
    if not rate_limiter.allow_request(client_id):
        remaining = rate_limiter.get_remaining(client_id)
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded", "remaining": remaining})
    return await call_next(request)


# Phase 4: Circuit breakers for external calls
llm_chat_circuit = AsyncCircuitBreaker("llm_chat", failure_threshold=3, recovery_timeout=30.0, default=None)
llm_embed_circuit = AsyncCircuitBreaker("llm_embed", failure_threshold=3, recovery_timeout=30.0, default=None)


# ============================================================================
# GLOBAL STATE
# ============================================================================

memory: Optional[RTMDKMemory] = None
from rtmdk.utils.lru_cache import LRUCache
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
        logger.warning("LM Studio not available at %s", LM_STUDIO_URL, exc_info=True)
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
        logger.warning("Embedding circuit open or failed, using fallback")
        rng = np.random.default_rng(hash(text) % 2**32)
        emb = rng.standard_normal(768).astype(np.float32) * 0.1
        embedder_cache.set(text, emb)
        return emb

    expected_dim = 768
    if len(embedding) != expected_dim:
        logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}. Resizing.")
        if len(embedding) > expected_dim:
            embedding = embedding[:expected_dim]
        else:
            embedding = np.pad(embedding, (0, expected_dim - len(embedding)), 'constant')

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
        logger.warning(f"Unknown preset '{preset_name}', falling back to 'production'")
        preset_fn = RTMDKConfig.production

    # Preset creates the base config, env vars override individual fields
    config = preset_fn()

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
            mem = RTMDKMemory.import_field(load_path, get_embedding, wal_path=load_path + ".wal")
            logger.info(f"Loaded memory from {load_path}: {len(mem.field.nodes)} nodes")
            return mem
        except Exception:
            logger.warning("Failed to load memory from %s", load_path, exc_info=True)
            import shutil
            backup_path = load_path + f".corrupted.{int(time.time())}"
            try:
                shutil.copy2(load_path, backup_path)
                os.remove(load_path)
            except Exception:
                pass

    save_path = _get_save_path(MEMORY_FILE)
    mem = RTMDKMemory(config=config, embedder=get_embedding, wal_path=save_path + ".wal")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mem.export_field(save_path)
        os.chmod(save_path, 0o600)  # Secure file permissions
        logger.info(f"Created new memory file at {save_path}")
    except Exception:
        logger.warning("Failed to create initial memory file", exc_info=True)

    return mem


def build_system_prompt(user_messages: List[ChatMessage], session_id: str) -> str:
    """Build system prompt with RTMDK context."""
    last_user = ""
    for msg in reversed(user_messages):
        if msg.role == "user":
            last_user = msg.content
            break

    ctx = {"rtmdk_context": ""}
    if last_user and memory:
        try:
            ctx = memory.load_memory_variables({"input": last_user, "session_id": session_id})
        except Exception:
            logger.warning("Memory query failed", exc_info=True)

    # Check for custom system prompt (env var file)
    prompt_file = os.getenv("RTMDK_SYSTEM_PROMPT_FILE")

    # Priority: env var file > env var text > config.system_prompt > None
    if prompt_file and os.path.exists(prompt_file):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
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
    if ctx["rtmdk_context"] and ctx["rtmdk_context"] not in ("No relevant memory.", "[]"):
        system_prompt += (
            f"\n\nRelevant memories:\n{ctx['rtmdk_context']}\n\n"
            "Use these memories to provide accurate, context-aware answers."
        )
    return system_prompt


def _get_save_path(base_path: str) -> str:
    """Select msgpack path if available, otherwise json."""
    try:
        import msgpack
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
        logger.debug(f"Auto-saved memory: {len(memory.field.nodes)} nodes to {save_path}")
    except Exception:
        logger.exception("Auto-save failed")


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup():
    global memory, lm_studio_available
    logger.info("Starting RTMDK Production API v8.0.0")
    logger.info(f"Memory file: {MEMORY_FILE}")
    logger.info(f"LM Studio URL: {LM_STUDIO_URL}")

    if ENABLE_LM_STUDIO:
        lm_studio_available = await check_lm_studio()

    memory = init_memory()
    global _memory_ref
    _memory_ref = memory
    asyncio.create_task(_auto_save_loop())

    logger.info(f"Server ready on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Memory nodes: {len(memory.field.nodes)}")


async def _auto_save_loop():
    """Background task that auto-saves memory periodically."""
    interval = int(os.getenv("RTMDK_AUTO_SAVE_INTERVAL", "60"))
    while True:
        await asyncio.sleep(interval)
        await run_sync(auto_save)


@app.on_event("shutdown")
async def shutdown():
    logger.info("RTMDK server shutting down...")
    if memory:
        # Gracefully stop background workers
        for task in memory.field._workers:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=10.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        memory.field._workers.clear()

        try:
            save_path = _get_save_path(MEMORY_FILE)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            await run_sync(memory.export_field, save_path)
            logger.info(f"Memory saved to {save_path} ({len(memory.field.nodes)} nodes)")
        except Exception:
            logger.exception("Failed to save memory on shutdown")


# ============================================================================
# OPENAI-COMPATIBLE ENDPOINTS
# ============================================================================

@app.get("/v1/models")
async def list_models():
    """List available models."""
    if chat_model:
        return {
            "object": "list",
            "data": [
                {"id": chat_model, "object": "model", "created": int(time.time()), "owned_by": "lm-studio"},
                {"id": "rtmdk", "object": "model", "created": int(time.time()), "owned_by": "rtmdk"},
            ]
        }
    return {"object": "list", "data": [{"id": "rtmdk", "object": "model", "created": int(time.time()), "owned_by": "rtmdk"}]}


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
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if last_user:
            try:
                await run_sync(memory.save_context,
                    {"input": last_user, "session_id": req.session_id},
                    {"output": ""}
                )
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
        raise HTTPException(status_code=503, detail="LM Studio unavailable (circuit open)")

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
                        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
                        if last_user:
                            await run_sync(memory.save_context,
                                {"input": last_user, "session_id": req.session_id},
                                {"output": "[streamed]"}
                            )
                    except Exception:
                        pass

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    data = resp.json()
    data["model"] = actual_model
    response_content = data["choices"][0]["message"]["content"]

    if memory and req.messages:
        try:
            last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            if last_user:
                await run_sync(memory.save_context,
                    {"input": last_user, "session_id": req.session_id},
                    {"output": response_content}
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
        data.append({
            "object": "embedding",
            "embedding": embedding.tolist(),
            "index": i,
        })
    return {
        "object": "list",
        "data": data,
        "model": req.model,
        "usage": {"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": sum(len(t.split()) for t in inputs)},
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "8.0.0",
        "lm_studio": lm_studio_available,
        "memory_nodes": len(memory.field.nodes) if memory else 0,
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    if not _PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501, detail="prometheus-client not installed")
    if memory:
        _metric_nodes.set(len(memory.field.nodes))
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# UX ENDPOINTS (from package)
# ============================================================================

def create_ux_router(memory_fn, config: Dict) -> 'APIRouter':
    """Import and return UX router from the package."""
    from rtmdk_server_ux import create_ux_router as _create_ux
    return _create_ux(memory_fn, config)


def create_dashboard_router(memory_fn, config: Dict) -> 'APIRouter':
    """Import and return Dashboard router from the package."""
    from rtmdk_dashboard_ui import create_dashboard_router as _create_dash
    return _create_dash(memory_fn, config)


_ux_config = {
    "RTMDK_BACKUP_DIR": os.path.join(os.path.expanduser("~"), ".rtmdk", "backups"),
    "RTMDK_SESSION_DIR": os.path.join(os.path.expanduser("~"), ".rtmdk", "sessions"),
    "RTMDK_CACHE_DIR": os.path.join(os.path.expanduser("~"), ".rtmdk", "embedding_cache"),
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
    print("    POST /v1/chat/completions  — Chat with memory")
    print("    POST /v1/embeddings        — Embeddings")
    print("    GET  /v1/models            — List models")
    print("    GET  /health               — Health check")
    print("    GET  /dashboard            — Web UI Dashboard")
    print("    GET  /api/models           — UX model selector")
    print("    POST /api/config           — Runtime configuration")
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
