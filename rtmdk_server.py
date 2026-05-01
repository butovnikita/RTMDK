"""
rtmdk_server.py
OpenAI-совместимый API-сервер для RTMDK Memory.

Zero-Config режим:
  python rtmdk_server.py

  Авто-детект:
    1. LM Studio на :12345 (если запущен)
    2. Память из ~/.rtmdk/memory.json (если существует)
    3. Сервер на :80801

Endpoints:
  POST /v1/chat/completions  — чат с инжекцией RTMDK-контекста
  POST /v1/embeddings        — эмбеддинги
  GET  /v1/models            — список моделей
  GET  /v1/memory/stats      — статистика памяти
  POST /v1/memory/imagine    — контрфактуальное воображение
  GET  /v1/memory/health     — здоровье поля
  POST /v1/memory/intervene  — do-интервенции
  POST /v1/memory/save       — сохранить контекст
  GET  /v1/memory/query      — запросить релевантную память
  POST /v1/memory/export     — экспорт состояния
  POST /v1/memory/import     — импорт состояния
  POST /v1/memory/clear      — очистить память
"""

import os
import sys
import json
import time
import uuid
import signal
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import asdict
from datetime import datetime, timezone
import asyncio

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from collections import OrderedDict
from pydantic import BaseModel, Field, field_validator
import uvicorn

# RTMDK imports
from rtmdk.memory.core import (
    RTMDKConfig, RTMDKMemory, ContextFormat,
    detect_modality, detect_tier,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Server settings
SERVER_HOST = os.getenv("RTMDK_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("RTMDK_PORT", "8080"))
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:12345/v1")
MEMORY_FILE = os.getenv("RTMDK_MEMORY_FILE", os.path.join(os.path.expanduser("~"), ".rtmdk", "memory.json"))
EMBED_MODEL = os.getenv("RTMDK_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
API_KEY = os.getenv("RTMDK_API_KEY", "rtmdk-local")
MEMORY_ENCRYPTION_KEY = os.getenv("RTMDK_MEMORY_ENCRYPTION_KEY", "")  # AES key for memory file encryption

# Security settings
ENABLE_API_AUTH = os.getenv("RTMDK_ENABLE_API_AUTH", "false").lower() == "true"  # Require API key
RATE_LIMIT_ENABLED = os.getenv("RTMDK_RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_PER_MIN = int(os.getenv("RTMDK_RATE_LIMIT_PER_MIN", "120"))
MAX_PAYLOAD_SIZE = int(os.getenv("RTMDK_MAX_PAYLOAD_SIZE", "1048576"))  # 1MB default
ALLOWED_ORIGINS = os.getenv("RTMDK_ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",")
ENABLE_LM_STUDIO = os.getenv("RTMDK_ENABLE_LM_STUDIO", "true").lower() == "true"
AUTO_SAVE_INTERVAL = int(os.getenv("RTMDK_AUTO_SAVE", "60"))  # seconds

# ============================================================================
# C2: STRUCTURED JSON LOGGING
# ============================================================================

class JSONFormatter(logging.Formatter):
    """C2: JSON structured log formatter for production logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        # Add trace_id if available in request state
        if hasattr(record, "trace_id") and record.trace_id:
            log_entry["trace_id"] = record.trace_id
        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Add extra fields
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry, ensure_ascii=False, default=str)


# Configure logging based on env var
LOG_FORMAT = os.getenv("RTMDK_LOG_FORMAT", "text").lower()
if LOG_FORMAT == "json":
    _json_handler = logging.StreamHandler(sys.stdout)
    _json_handler.setFormatter(JSONFormatter())
    logging.basicConfig(
        level=logging.INFO,
        handlers=[_json_handler],
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
logger = logging.getLogger("rtmdk_server")

# ============================================================================
# APP INITIALIZATION
# ============================================================================

app = FastAPI(
    title="RTMDK Memory API",
    description="OpenAI-compatible API with Resonance-Topological Memory",
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
# SECURITY: API Key Authentication Middleware
# ============================================================================

@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Enforce API key authentication and rate limiting."""
    # Skip auth for health and model endpoints (needed for LM Studio detection)
    skip_auth_paths = ["/health", "/v1/models", "/docs", "/openapi.json", "/redoc", "/dashboard"]
    if request.url.path in skip_auth_paths or request.url.path.startswith("/api/"):
        return await call_next(request)
    
    # Check payload size
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_PAYLOAD_SIZE:
        return JSONResponse(
            status_code=413,
            content={"error": "Payload too large"}
        )
    
    # Check API key if enabled
    if ENABLE_API_AUTH:
        auth_header = request.headers.get("authorization", "")
        api_key = auth_header.replace("Bearer ", "").replace("bearer ", "") if auth_header else ""

        # Also check x-api-key header for compatibility
        if not api_key:
            api_key = request.headers.get("x-api-key", "")

        if not api_key or api_key != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized. Provide valid API key."}
            )

    # Check rate limit
    if RATE_LIMIT_ENABLED and _rate_limiter:
        client_id = request.client.host if request.client else "unknown"
        if not _rate_limiter.allow_request(client_id):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."}
            )

    return await call_next(request)

# Global state
memory: Optional[RTMDKMemory] = None
embedder_cache: OrderedDict[str, np.ndarray] = OrderedDict()
EMBEDDER_CACHE_MAX_SIZE = 10000  # Max 10K entries to prevent memory exhaustion
_import_lock = asyncio.Lock()  # Prevent race condition on memory import
lm_studio_available: bool = False
chat_model: Optional[str] = None
auto_save_task = None

# Rate limiter (from production module)
_rate_limiter = None
try:
    from rtmdk.production.rate_limiter import RateLimiter
    _rate_limiter = RateLimiter(max_per_minute=RATE_LIMIT_PER_MIN)
except ImportError:
    logger.warning("Rate limiter not available, rate limiting disabled")

# C3: Prometheus metrics counters
_metrics_queries_total = 0
_metrics_consolidations_total = 0
_metrics_query_latencies: List[float] = []  # in ms
_metrics_field_health: int = 0  # 0=stable, 1=degraded, 2=critical

# ============================================================================
# PYDANTIC MODELS (OpenAI-compatible)
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

    @field_validator("input")
    @classmethod
    def validate_input(cls, v):
        if isinstance(v, list):
            if len(v) > 100:
                raise ValueError("Maximum 100 inputs per request")
            for i, item in enumerate(v):
                if len(item) > 10000:
                    raise ValueError(f"Input {i} exceeds max length (10000)")
        return v

class ImagineRequest(BaseModel):
    query: str
    intervention: Dict[str, float]
    session_id: str = "default"

class InterveneRequest(BaseModel):
    node_id: str
    text: str

class SaveContextRequest(BaseModel):
    input: str
    output: str
    session_id: str = "default"

class QueryMemoryRequest(BaseModel):
    query: str
    session_id: str = "default"
    top_k: int = 5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Manages model discovery and caching from LM Studio."""
    
    def __init__(self, lm_studio_url: str):
        self.lm_studio_url = lm_studio_url
        self._chat_models: List[Dict] = []
        self._embedder_models: List[Dict] = []
        self._all_models: List[Dict] = []
        self._last_refresh: float = 0
        self._refresh_interval: int = 60  # Refresh every 60 seconds
    
    def refresh_models(self) -> bool:
        """Fetch models from LM Studio."""
        import requests
        try:
            resp = requests.get(f"{self.lm_studio_url}/models", timeout=5)
            data = resp.json()
            models = data.get("data", [])
            
            self._all_models = models
            self._chat_models = []
            self._embedder_models = []
            
            # Separate by capabilities
            for model in models:
                model_id = model.get("id", "")
                # Simple heuristic: embedding models have "embed" in name
                if "embed" in model_id.lower():
                    self._embedder_models.append(model)
                else:
                    self._chat_models.append(model)
            
            self._last_refresh = time.time()
            logger.info(f"Model refresh: {len(self._chat_models)} chat, {len(self._embedder_models)} embedder")
            return True
        except Exception as e:
            logger.warning(f"Model refresh failed: {e}")
            return False
    
    @property
    def chat_models(self) -> List[Dict]:
        if not self._chat_models or (time.time() - self._last_refresh > self._refresh_interval):
            self.refresh_models()
        return self._chat_models
    
    @property
    def embedder_models(self) -> List[Dict]:
        if not self._embedder_models or (time.time() - self._last_refresh > self._refresh_interval):
            self.refresh_models()
        return self._embedder_models
    
    @property
    def all_models(self) -> List[Dict]:
        if not self._all_models or (time.time() - self._last_refresh > self._refresh_interval):
            self.refresh_models()
        return self._all_models
    
    @property
    def default_chat_model(self) -> str:
        models = self.chat_models
        return models[0]["id"] if models else "rtmdk"
    
    @property
    def default_embedder_model(self) -> str:
        models = self.embedder_models
        return models[0]["id"] if models else EMBED_MODEL


# Global model manager
model_manager: Optional[ModelManager] = None


def check_lm_studio() -> bool:
    """Check if LM Studio is available and initialize model manager."""
    global model_manager, chat_model
    try:
        model_manager = ModelManager(LM_STUDIO_URL)
        if model_manager.refresh_models():
            chat_model = model_manager.default_chat_model
            logger.info(f"LM Studio detected: {chat_model}")
            return True
    except Exception as e:
        logger.warning(f"LM Studio init failed: {e}")
    logger.warning("LM Studio not available at %s", LM_STUDIO_URL)
    return False


def get_embedding(text: str, model: str = None) -> np.ndarray:
    """Get embedding from LM Studio or cache."""
    if text in embedder_cache:
        return embedder_cache[text]

    import requests
    embedder_model = model or model_manager.default_embedder_model if model_manager else EMBED_MODEL
    
    max_retries = 3
    base_timeout = int(os.getenv("RTMDK_LM_STUDIO_TIMEOUT", "30"))
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{LM_STUDIO_URL}/embeddings",
                json={"model": embedder_model, "input": text},
                timeout=base_timeout,
            )
            data = resp.json()
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            
            # Validate embedding dimension
            expected_dim = 768  # Default, should match config
            if len(embedding) != expected_dim:
                logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}. Resizing.")
                if len(embedding) > expected_dim:
                    embedding = embedding[:expected_dim]
                else:
                    embedding = np.pad(embedding, (0, expected_dim - len(embedding)), 'constant')
            
            embedder_cache[text] = embedding
            # H1: LRU eviction to prevent memory exhaustion
            if len(embedder_cache) > EMBEDDER_CACHE_MAX_SIZE:
                embedder_cache.popitem(last=False)
            return embedding
        except requests.exceptions.Timeout:
            logger.warning(f"Embedding timeout on attempt {attempt+1}/{max_retries}")
            if attempt == max_retries - 1:
                break
            time.sleep(1 * (attempt + 1))
        except Exception as e:
            logger.warning(f"Embedding error: {e}, using fallback")
            break

    np.random.seed(hash(text) % 2**32)
    emb = np.random.randn(768).astype(np.float32) * 0.1
    embedder_cache[text] = emb
    return emb


def init_memory() -> RTMDKMemory:
    """Initialize or load RTMDK memory.
    
    Configuration is loaded from preset (RTMDK_PRESET env var, default "local")
    with individual field overrides via RTMDK_* env vars.
    """
    preset_name = os.getenv("RTMDK_PRESET", "local")
    preset_fn = getattr(RTMDKConfig, preset_name, None)
    if preset_fn is None:
        logger.warning(f"Unknown preset '{preset_name}', falling back to 'local'")
        preset_fn = RTMDKConfig.local

    # Preset creates the base config, env vars override individual fields
    config = preset_fn()

    # Override top_k from env if set
    env_top_k = os.getenv("RTMDK_TOP_K")
    if env_top_k:
        config.top_k = int(env_top_k)
        logger.info(f"  top_k overridden from env: {config.top_k}")

    logger.info(f"Memory config preset: {preset_name}")
    logger.info(f"  latent_dim={config.latent_dim}, decay={config.decay_rate}")
    logger.info(f"  tension={config.tension_threshold}, top_k={config.top_k}")
    logger.info(f"  cross_modal={config.cross_modal}, self_healing={config.self_healing}")

    # Try to load existing memory
    if os.path.exists(MEMORY_FILE):
        # Quick health check before attempting import
        try:
            file_size = os.path.getsize(MEMORY_FILE)
            import json as _json
            _test = _json.load(open(MEMORY_FILE, encoding='utf-8'))
            _nodes = _test.get('nodes', [])
            logger.info(f"Memory file health check: {file_size/1024:.0f}KB, {len(_nodes)} nodes, keys={list(_test.keys())}")
        except Exception as he:
            logger.warning(f"Memory file health check failed: {he}")

        try:
            mem = RTMDKMemory.import_field(MEMORY_FILE, get_embedding)
            logger.info(f"Loaded memory from {MEMORY_FILE}: {len(mem.field.nodes)} nodes")
            # Apply context_format override from env/preset if different from file
            env_fmt = os.getenv("RTMDK_CONTEXT_FORMAT")
            if env_fmt:
                from rtmdk.memory.core import ContextFormat
                mem.config.context_format = ContextFormat(env_fmt)
                mem.field.stats["context_format"] = env_fmt
                logger.info(f"  context_format overridden from env: {env_fmt}")
            return mem
        except (json.JSONDecodeError, ValueError, FileNotFoundError) as e:
            # File corruption — safe to backup and recreate
            logger.warning(f"Memory file corrupted: {e}")
            import shutil
            backup_path = MEMORY_FILE + f".corrupted.{int(time.time())}"
            try:
                shutil.copy2(MEMORY_FILE, backup_path)
                logger.info(f"Backed up corrupted memory file to: {backup_path}")
                os.remove(MEMORY_FILE)
                logger.warning(f"Deleted corrupted memory file. Starting with fresh memory.")
            except Exception as backup_err:
                logger.error(f"Failed to backup corrupted memory: {backup_err}")
        except Exception as e:
            # Import error (e.g., embedder failure) — DO NOT delete the file
            logger.error(f"Failed to import memory from {MEMORY_FILE}: {e}")
            logger.warning(f"Memory file preserved. Fix the error and restart.")
            logger.warning(f"Starting with fresh memory — your data is safe in {MEMORY_FILE}")

    mem = RTMDKMemory(config=config, embedder=get_embedding)
    
    # Save initial empty memory file so it exists for next startup
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        mem.export_field(MEMORY_FILE)
        logger.info(f"Created new memory file at {MEMORY_FILE}")
    except Exception as e:
        logger.warning(f"Failed to create initial memory file: {e}")
    
    logger.info("Initialized new RTMDK memory")
    return mem


# ═══════════════════════════════════════════════════════════
# UX Endpoints & Dashboard Integration
# ═══════════════════════════════════════════════════════════

from rtmdk_server_ux import create_ux_router
from rtmdk_dashboard_ui import create_dashboard_router

_ux_config = {
    "RTMDK_BACKUP_DIR": os.path.join(os.path.expanduser("~"), ".rtmdk", "backups"),
    "RTMDK_SESSION_DIR": os.path.join(os.path.expanduser("~"), ".rtmdk", "sessions"),
    "RTMDK_CACHE_DIR": os.path.join(os.path.expanduser("~"), ".rtmdk", "embedding_cache"),
    "RTMDK_CACHE_MAX_SIZE": "100000",
}
app.include_router(create_ux_router(lambda: memory, _ux_config))
app.include_router(create_dashboard_router(lambda: memory, _ux_config))


def auto_save():
    """Auto-save memory to file."""
    if memory:
        try:
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            memory.export_field(MEMORY_FILE)
            logger.debug(f"Auto-saved memory: {len(memory.field.nodes)} nodes")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")


def build_system_prompt(user_messages: List[ChatMessage], session_id: str) -> str:
    """Build system prompt with RTMDK context."""
    # Get last user message for memory query
    last_user = ""
    for msg in reversed(user_messages):
        if msg.role == "user":
            last_user = msg.content
            break

    # Query RTMDK memory
    ctx = {"rtmdk_context": ""}
    if last_user and memory:
        try:
            ctx = memory.load_memory_variables({"input": last_user, "session_id": session_id})
        except Exception as e:
            logger.warning(f"Memory query failed: {e}")

    # Check for custom system prompt (env var file)
    prompt_file = os.getenv("RTMDK_SYSTEM_PROMPT_FILE")

    # Priority: env var file > env var text > config.system_prompt > None
    if prompt_file and os.path.exists(prompt_file):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                base_prompt = f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read prompt file: {e}")
            base_prompt = memory.config.system_prompt if memory else None
    else:
        # Check if env var overrides config
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
            "\n\nRelevant memories from previous conversations:\n"
            f"{ctx['rtmdk_context']}"
            "\n\nUse these memories to provide accurate, context-aware answers. "
            "If a memory is marked as [HYPOTHETICAL], treat it as a hypothetical scenario."
        )

    return system_prompt


# ============================================================================
# OPENAI-COMPATIBLE ENDPOINTS
# ============================================================================

@app.get("/v1/models")
async def list_models():
    """List available models from LM Studio in OpenAI format."""
    global model_manager
    
    if model_manager and model_manager.all_models:
        # Return all models from LM Studio in OpenAI format
        return {
            "object": "list",
            "data": model_manager.all_models
        }
    else:
        # Fallback: return minimal model list
        return {
            "object": "list",
            "data": [
                {
                    "id": chat_model or "rtmdk",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "lm-studio"
                }
            ]
        }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Chat completions with RTMDK memory context."""
    logger.info(f"Chat request: stream={req.stream}, model={req.model}, messages={len(req.messages)}")

    if not lm_studio_available:
        logger.error("LM Studio not available!")
        raise HTTPException(
            status_code=503,
            detail="LM Studio not available. Start LM Studio and enable server on port 12345."
        )

    import requests

    # Build system prompt with memory context
    system_prompt = build_system_prompt(req.messages, req.session_id)

    # Build messages for LM Studio (only add system message if prompt exists)
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
                memory.save_context(
                    {"input": last_user, "session_id": req.session_id},
                    {"output": ""}  # Will be updated after response
                )
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

    # Call LM Studio
    lm_timeout = int(os.getenv("RTMDK_LM_STUDIO_TIMEOUT", "120"))
    # Use model from request, fallback to default
    request_model = req.model if hasattr(req, 'model') and req.model and req.model != "rtmdk" else None
    actual_model = request_model or chat_model or "local-model"
    
    max_retries = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{LM_STUDIO_URL}/chat/completions",
                json={
                    "model": actual_model,
                    "messages": messages,
                    "temperature": req.temperature,
                    "max_tokens": req.max_tokens,
                    "stream": req.stream,
                    "top_p": req.top_p,
                    "frequency_penalty": req.frequency_penalty,
                    "presence_penalty": req.presence_penalty,
                },
                timeout=lm_timeout,
                stream=req.stream,
            )
            last_error = None
            break  # Success
        except requests.exceptions.Timeout:
            last_error = f"LM Studio timeout after {lm_timeout}s (attempt {attempt+1}/{max_retries})"
            logger.warning(last_error)
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
        except requests.exceptions.ConnectionError as e:
            last_error = f"LM Studio connection error: {e}"
            logger.warning(last_error)
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
        except requests.exceptions.RequestException as e:
            last_error = f"LM Studio request failed: {e}"
            logger.warning(last_error)
            break  # Don't retry on other errors

    if last_error:
        raise HTTPException(status_code=502, detail=last_error)

    if req.stream:
        logger.info(f"Streaming response enabled, timeout={lm_timeout}s")
        async def stream_generator():
            chunk_count = 0
            total_chars = 0
            collected_text = []  # Collect all text chunks for memory save

            try:
                # Check if response is actually streaming
                if not hasattr(resp, 'iter_lines'):
                    logger.warning("Response doesn't support streaming, falling back")
                    data = resp.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    yield f'data: {json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]})}\n\n'
                    yield 'data: [DONE]\n\n'
                    return

                for chunk in resp.iter_lines(chunk_size=1):
                    if chunk:
                        line = chunk.decode("utf-8", errors='replace')
                        if line.startswith("data: "):
                            chunk_count += 1
                            total_chars += len(line)
                            yield f"{line}\n\n"
                            
                            # Extract text content for memory save
                            try:
                                data_part = line[6:]  # Remove "data: " prefix
                                if data_part.strip() and data_part.strip() != '[DONE]':
                                    chunk_data = json.loads(data_part)
                                    choices = chunk_data.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            collected_text.append(content)
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass  # Skip malformed chunks
                        elif line.strip() == '[DONE]':
                            yield 'data: [DONE]\n\n'
                            break

            except Exception as e:
                logger.error(f"Streaming error after {chunk_count} chunks: {e}")
            finally:
                logger.info(f"Streaming completed: {chunk_count} chunks, {total_chars} chars")
                # Save final response to memory
                if memory and collected_text:
                    try:
                        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
                        if last_user:
                            memory.save_context(
                                {"input": last_user, "session_id": req.session_id},
                                {"output": "".join(collected_text)}
                            )
                    except Exception as e:
                        logger.error(f"Memory save error: {e}")

        return StreamingResponse(stream_generator(), media_type="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        })

    data = resp.json()
    # Ensure response has correct model name
    data["model"] = actual_model
    response_content = data["choices"][0]["message"]["content"]

    # Update memory with response
    if memory and req.messages:
        try:
            last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
            if last_user:
                memory.save_context(
                    {"input": last_user, "session_id": req.session_id},
                    {"output": response_content}
                )
        except Exception as e:
            logger.warning(f"Memory update failed: {e}")

    return data


@app.get("/v1/test/streaming")
async def test_streaming():
    """Diagnostic endpoint to test LM Studio streaming capability."""
    if not lm_studio_available:
        return {"streaming": False, "error": "LM Studio not available"}

    import requests
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": chat_model or "local-model",
                "messages": [{"role": "user", "content": "Say 'test' in one word."}],
                "stream": True,
                "max_tokens": 10,
            },
            timeout=30,
            stream=True,
        )

        chunks = []
        content_type = resp.headers.get('Content-Type', 'unknown')
        is_chunked = resp.headers.get('Transfer-Encoding', '') == 'chunked'
        
        logger.info(f"LM Studio response: Content-Type={content_type}, Chunked={is_chunked}")

        for line in resp.iter_lines():
            if line:
                decoded = line.decode('utf-8', errors='replace')
                chunks.append(decoded)
                if len(chunks) >= 10:
                    break

        return {
            "streaming": True,
            "content_type": content_type,
            "is_chunked": is_chunked,
            "chunks_received": len(chunks),
            "first_chunk": chunks[0] if chunks else None,
            "sample_chunks": chunks[:5],
        }
    except Exception as e:
        logger.error(f"Streaming test failed: {e}")
        return {"streaming": False, "error": str(e)}


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    """Create embeddings using specified model."""
    # Use model from request if provided
    embedder_model = req.model if hasattr(req, 'model') and req.model else None
    
    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = []
    for i, text in enumerate(inputs):
        embedding = get_embedding(text, model=embedder_model)
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


# ============================================================================
# RTMDK MEMORY ENDPOINTS
# ============================================================================

@app.get("/v1/memory/stats")
async def get_memory_stats():
    """Get RTMDK memory statistics."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    stats = memory.get_stats()
    return stats


@app.get("/v1/memory/health")
async def get_memory_health():
    """Get memory field health."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    health = memory.get_field_health()
    health["nodes"] = memory.field.stats.get("active_nodes", 0)
    health["causal_edges"] = memory.field.stats.get("causal_edges", 0)
    health["contradictions"] = memory.field.stats.get("contradictions", 0)
    return health


@app.post("/v1/memory/imagine")
async def imagine_counterfactual(req: ImagineRequest):
    """Generate counterfactual scenarios."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    results = memory.imagine_counterfactual(req.query, req.intervention)
    return {"scenarios": [r for r in results]}


@app.post("/v1/memory/intervene")
async def do_intervention(req: InterveneRequest):
    """Apply causal intervention do(X=x)."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    memory.do_intervention(req.node_id, req.text)
    return {"status": "ok", "node_id": req.node_id}


@app.post("/v1/memory/save")
async def save_context(req: SaveContextRequest):
    """Save context to memory."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    memory.save_context(
        {"input": req.input, "session_id": req.session_id},
        {"output": req.output}
    )
    return {"status": "ok", "nodes": len(memory.field.nodes)}


@app.post("/v1/memory/query")
async def query_memory(req: QueryMemoryRequest):
    """Query relevant memory."""
    global _metrics_queries_total, _metrics_query_latencies
    _metrics_queries_total += 1
    t0 = time.time()
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    ctx = memory.load_memory_variables({"input": req.query, "session_id": req.session_id})
    latency_ms = (time.time() - t0) * 1000
    _metrics_query_latencies.append(latency_ms)
    # Auto-trim to prevent unbounded growth (not just on /metrics calls)
    if len(_metrics_query_latencies) > 2000:
        _metrics_query_latencies[:] = _metrics_query_latencies[-1000:]
    return {"context": ctx["rtmdk_context"]}


@app.post("/v1/memory/export")
async def export_memory():
    """Export memory state."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    memory.export_field(MEMORY_FILE)
    return {"status": "ok", "file": MEMORY_FILE, "nodes": len(memory.field.nodes)}


@app.post("/v1/memory/import")
async def import_memory():
    """Import memory state. Uses lock to prevent race condition."""
    global memory
    if not os.path.exists(MEMORY_FILE):
        raise HTTPException(status_code=404, detail=f"Memory file not found: {MEMORY_FILE}")
    async with _import_lock:
        memory = RTMDKMemory.import_field(MEMORY_FILE, get_embedding)
    return {"status": "ok", "nodes": len(memory.field.nodes)}


@app.post("/v1/memory/clear")
async def clear_memory():
    """Clear all memory."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    memory.clear()
    return {"status": "ok"}


@app.get("/v1/memory/causal")
async def get_causal_summary():
    """Get causal graph summary."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    return memory.get_causal_summary()


@app.get("/v1/memory/contradictions")
async def get_contradictions():
    """Get detected contradictions."""
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    contradictions = memory.get_contradictions()
    return {"contradictions": [c.to_dict() for c in contradictions]}


@app.get("/health")
async def health_check():
    """Server health check."""
    return {
        "status": "ok",
        "version": "8.0.0",
        "lm_studio": lm_studio_available,
        "memory_nodes": len(memory.field.nodes) if memory else 0,
    }


# ============================================================================
# C3: PROMETHEUS /metrics ENDPOINT
# ============================================================================

def _compute_field_health() -> int:
    """C3: Compute field health: 0=stable, 1=degraded, 2=critical."""
    if not memory or not memory.field:
        return 0
    health = memory.field.stats.get("field_health", "stable")
    health_map = {"stable": 0, "degraded": 1, "critical": 2, "healing": 1}
    return health_map.get(str(health), 0)


def _histogram_to_prometheus(buckets: List[float], name: str) -> str:
    """Convert latency list to Prometheus histogram format."""
    if not buckets:
        return (
            f"{name}_bucket{{le=\"10\"}} 0\n"
            f"{name}_bucket{{le=\"50\"}} 0\n"
            f"{name}_bucket{{le=\"100\"}} 0\n"
            f"{name}_bucket{{le=\"500\"}} 0\n"
            f"{name}_bucket{{le=\"1000\"}} 0\n"
            f"{name}_bucket{{le=\"+Inf\"}} 0\n"
            f"{name}_sum 0.0\n"
            f"{name}_count 0\n"
        )
    bounds = [10, 50, 100, 500, 1000]
    lines = []
    cumulative = 0
    for bound in bounds:
        cumulative += sum(1 for v in buckets if v <= bound)
        lines.append(f"{name}_bucket{{le=\"{bound}\"}} {cumulative}")
    cumulative = len(buckets)
    lines.append(f"{name}_bucket{{le=\"+Inf\"}} {cumulative}")
    lines.append(f"{name}_sum {sum(buckets):.2f}")
    lines.append(f"{name}_count {cumulative}")
    return "\n".join(lines)


@app.get("/metrics")
async def prometheus_metrics():
    """C3: Prometheus text exposition format metrics."""
    global _metrics_queries_total, _metrics_consolidations_total
    global _metrics_query_latencies, _metrics_field_health

    nodes_total = len(memory.field.nodes) if memory and memory.field else 0
    field_health = _compute_field_health()
    _metrics_field_health = field_health

    # Trim latency history to last 1000 entries
    _metrics_query_latencies = _metrics_query_latencies[-1000:]

    lines = [
        "# HELP rtmdk_nodes_total Total number of memory nodes.",
        "# TYPE rtmdk_nodes_total gauge",
        f"rtmdk_nodes_total {nodes_total}",
        "",
        "# HELP rtmdk_queries_total Total number of queries.",
        "# TYPE rtmdk_queries_total counter",
        f"rtmdk_queries_total {_metrics_queries_total}",
        "",
        "# HELP rtmdk_consolidations_total Total number of consolidations.",
        "# TYPE rtmdk_consolidations_total counter",
        f"rtmdk_consolidations_total {_metrics_consolidations_total}",
        "",
        "# HELP rtmdk_query_latency_ms Query latency histogram.",
        "# TYPE rtmdk_query_latency_ms histogram",
        _histogram_to_prometheus(_metrics_query_latencies, "rtmdk_query_latency_ms"),
        "",
        "# HELP rtmdk_field_health Field health gauge (0=stable, 1=degraded, 2=critical).",
        "# TYPE rtmdk_field_health gauge",
        f"rtmdk_field_health {field_health}",
    ]
    return "\n".join(lines) + "\n"


# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup():
    global memory, lm_studio_available

    logger.info("Starting RTMDK Memory API v8.0.0")
    logger.info(f"Memory file: {MEMORY_FILE}")
    logger.info(f"LM Studio URL: {LM_STUDIO_URL}")

    # Check LM Studio
    if ENABLE_LM_STUDIO:
        lm_studio_available = check_lm_studio()

    # Initialize memory
    memory = init_memory()

    # Start auto-save background task
    global auto_save_task
    auto_save_task = asyncio.create_task(_auto_save_loop())

    logger.info(f"Server ready on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Memory nodes: {len(memory.field.nodes)}")


async def _auto_save_loop():
    """Background task that auto-saves memory periodically."""
    interval = int(os.getenv("RTMDK_AUTO_SAVE_INTERVAL", "60"))
    while True:
        await asyncio.sleep(interval)
        auto_save()


@app.on_event("shutdown")
async def shutdown():
    """C4: Save memory and clean up on shutdown."""
    logger.info("RTMDK server shutting down...")
    # H2: Cancel auto-save task to prevent write after deallocation
    global auto_save_task
    if auto_save_task and not auto_save_task.done():
        auto_save_task.cancel()
        try:
            await auto_save_task
        except asyncio.CancelledError:
            pass
    if memory:
        try:
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            memory.export_field(MEMORY_FILE)
            logger.info(f"Memory saved to {MEMORY_FILE} ({len(memory.field.nodes)} nodes)")
        except Exception as e:
            logger.error(f"Failed to save memory on shutdown: {e}")
    logger.info("RTMDK server shutdown complete")


# C4: Signal handlers for graceful shutdown
def _graceful_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    global memory
    if memory:
        try:
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            memory.export_field(MEMORY_FILE)
            logger.info(f"Memory saved on signal {signum}")
        except Exception as e:
            logger.error(f"Memory save failed on signal {signum}: {e}")
    logger.info("Graceful shutdown complete")
    # Don't call sys.exit(0) - let uvicorn handle shutdown
    raise KeyboardInterrupt()


def _register_signal_handlers():
    """Register signal handlers for graceful shutdown."""
    if os.name != "nt":
        # SIGTERM not available on Windows
        signal.signal(signal.SIGTERM, _graceful_shutdown)
        signal.signal(signal.SIGINT, _graceful_shutdown)
    else:
        # On Windows, SIGINT works but we need to handle it carefully
        try:
            signal.signal(signal.SIGINT, _graceful_shutdown)
        except (ValueError, OSError):
            pass  # Signal registration failed, uvicorn will handle shutdown


# ============================================================================
# SILLY TAVERN COMPATIBILITY (Text Completions API)
# ============================================================================

from rtmdk_sillytavern_compat import create_sillytavern_router
app.include_router(create_sillytavern_router(
    lambda: memory, _ux_config, lambda: lm_studio_available,
    lambda: chat_model, LM_STUDIO_URL
))

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  RTMDK Memory API v8.0.0")
    print("  OpenAI-compatible endpoint with Resonance-Topological Memory")
    print("=" * 60)
    print()
    print(f"  Server: http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"  Memory: {MEMORY_FILE}")
    print(f"  LM Studio: {LM_STUDIO_URL}")
    print()
    print("  Endpoints:")
    print("    POST /v1/chat/completions  — Chat with memory")
    print("    POST /v1/embeddings        — Embeddings")
    print("    GET  /v1/models            — List models")
    print("    GET  /v1/memory/stats      — Memory statistics")
    print("    GET  /v1/memory/health     — Field health")
    print("    POST /v1/memory/imagine    — Counterfactual scenarios")
    print("    POST /v1/memory/intervene  — Causal intervention")
    print("    POST /v1/memory/save       — Save context")
    print("    POST /v1/memory/query      — Query memory")
    print("    GET  /v1/memory/causal     — Causal summary")
    print("    GET  /v1/memory/contradictions — Contradictions")
    print("    POST /v1/memory/export     — Export state")
    print("    POST /v1/memory/import     — Import state")
    print("    POST /v1/memory/clear      — Clear memory")
    print()
    print("  IDE Integration:")
    print("    Cursor/Continue/Aider: set base URL to http://localhost:8080/v1")
    print("    API Key: rtmdk-local")
    print(f"    Dashboard: http://{SERVER_HOST}:{SERVER_PORT}/dashboard")
    print("-" * 60)

    # Register signal handlers for graceful shutdown
    _register_signal_handlers()

    uvicorn.run(
        "rtmdk_server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
