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
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# RTMDK package imports
from rtmdk.memory.core import RTMDKMemory, RTMDKConfig


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

# ============================================================================
# LOGGING
# ============================================================================

log_level = getattr(logging, os.getenv("RTMDK_LOG_LEVEL", "INFO"))
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
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


# ============================================================================
# GLOBAL STATE
# ============================================================================

memory: Optional[RTMDKMemory] = None
embedder_cache: Dict[str, np.ndarray] = {}
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

def check_lm_studio() -> bool:
    """Check if LM Studio is available."""
    import requests
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/models", timeout=3)
        global chat_model
        models = resp.json().get("data", [])
        if models:
            chat_model = models[0]["id"]
            logger.info(f"LM Studio detected: {chat_model}")
            return True
    except Exception:
        pass
    logger.warning("LM Studio not available at %s", LM_STUDIO_URL)
    return False


def get_embedding(text: str, model: str = None) -> np.ndarray:
    """Get embedding from LM Studio or cache."""
    if text in embedder_cache:
        return embedder_cache[text]

    import requests
    embedder_model = model or EMBED_MODEL

    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/embeddings",
            json={"model": embedder_model, "input": text},
            timeout=30,
        )
        data = resp.json()
        embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)

        expected_dim = 768
        if len(embedding) != expected_dim:
            logger.warning(f"Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}. Resizing.")
            if len(embedding) > expected_dim:
                embedding = embedding[:expected_dim]
            else:
                embedding = np.pad(embedding, (0, expected_dim - len(embedding)), 'constant')

        embedder_cache[text] = embedding
        return embedding
    except Exception as e:
        logger.warning(f"Embedding error: {e}, using fallback")
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(768).astype(np.float32) * 0.1
        embedder_cache[text] = emb
        return emb


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

    if os.path.exists(MEMORY_FILE):
        try:
            mem = RTMDKMemory.import_field(MEMORY_FILE, get_embedding)
            logger.info(f"Loaded memory from {MEMORY_FILE}: {len(mem.field.nodes)} nodes")
            return mem
        except Exception as e:
            logger.warning(f"Failed to load memory from {MEMORY_FILE}: {e}")
            import shutil
            backup_path = MEMORY_FILE + f".corrupted.{int(time.time())}"
            try:
                shutil.copy2(MEMORY_FILE, backup_path)
                os.remove(MEMORY_FILE)
            except Exception:
                pass

    mem = RTMDKMemory(config=config, embedder=get_embedding)
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        mem.export_field(MEMORY_FILE)
        os.chmod(MEMORY_FILE, 0o600)  # Secure file permissions
        logger.info(f"Created new memory file at {MEMORY_FILE}")
    except Exception as e:
        logger.warning(f"Failed to create initial memory file: {e}")

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
        except Exception as e:
            logger.warning(f"Memory query failed: {e}")

    # Check for custom system prompt (env var or file)
    prompt_file = os.getenv("RTMDK_SYSTEM_PROMPT_FILE")
    custom_prompt = os.getenv("RTMDK_SYSTEM_PROMPT")

    if prompt_file and os.path.exists(prompt_file):
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                base_prompt = f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to read prompt file: {e}")
            base_prompt = "You are a helpful assistant with long-term memory powered by RTMDK."
    elif custom_prompt:
        base_prompt = custom_prompt
    else:
        base_prompt = "You are a helpful assistant with long-term memory powered by RTMDK."

    system_prompt = base_prompt
    if ctx["rtmdk_context"] and ctx["rtmdk_context"] not in ("No relevant memory.", "[]"):
        system_prompt += (
            f"\n\nRelevant memories:\n{ctx['rtmdk_context']}\n\n"
            "Use these memories to provide accurate, context-aware answers."
        )
    return system_prompt


def auto_save():
    """Auto-save memory to file."""
    if memory:
        try:
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            memory.export_field(MEMORY_FILE)
            logger.debug(f"Auto-saved memory: {len(memory.field.nodes)} nodes")
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")


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
        lm_studio_available = check_lm_studio()

    memory = init_memory()
    asyncio.create_task(_auto_save_loop())

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
    logger.info("RTMDK server shutting down...")
    if memory:
        try:
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            memory.export_field(MEMORY_FILE)
            logger.info(f"Memory saved to {MEMORY_FILE} ({len(memory.field.nodes)} nodes)")
        except Exception as e:
            logger.error(f"Failed to save memory on shutdown: {e}")


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

    import requests
    system_prompt = build_system_prompt(req.messages, req.session_id)
    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # Save user input to memory
    if memory and req.messages:
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        if last_user:
            try:
                memory.save_context(
                    {"input": last_user, "session_id": req.session_id},
                    {"output": ""}
                )
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

    lm_timeout = int(os.getenv("RTMDK_LM_STUDIO_TIMEOUT", "120"))
    request_model = req.model if req.model and req.model != "rtmdk" else None
    actual_model = request_model or chat_model or "local-model"

    try:
        resp = requests.post(
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
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))

    if req.stream:
        async def stream_generator():
            try:
                for chunk in resp.iter_lines():
                    if chunk:
                        line = chunk.decode("utf-8", errors='replace')
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
            finally:
                if memory:
                    try:
                        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
                        if last_user:
                            memory.save_context(
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
                memory.save_context(
                    {"input": last_user, "session_id": req.session_id},
                    {"output": response_content}
                )
        except Exception as e:
            logger.warning(f"Memory update failed: {e}")

    return data


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    """Create embeddings."""
    inputs = req.input if isinstance(req.input, list) else [req.input]
    data = []
    for i, text in enumerate(inputs):
        embedding = get_embedding(text)
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
