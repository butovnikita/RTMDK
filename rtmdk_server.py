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

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# RTMDK imports
from rtmdk_memory_v8 import (
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
ENABLE_LM_STUDIO = os.getenv("RTMDK_ENABLE_LM_STUDIO", "true").lower() == "true"
AUTO_SAVE_INTERVAL = int(os.getenv("RTMDK_AUTO_SAVE", "60"))  # seconds

# Logging
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
memory: Optional[RTMDKMemory] = None
embedder_cache: Dict[str, np.ndarray] = {}
lm_studio_available: bool = False
chat_model: Optional[str] = None
auto_save_task = None

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


def get_embedding(text: str) -> np.ndarray:
    """Get embedding from LM Studio or cache."""
    if text in embedder_cache:
        return embedder_cache[text]

    import requests
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/embeddings",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30,
        )
        data = resp.json()
        embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
        embedder_cache[text] = embedding
        return embedding
    except Exception as e:
        logger.warning(f"Embedding error: {e}, using fallback")
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(768).astype(np.float32) * 0.1
        embedder_cache[text] = emb
        return emb


def init_memory() -> RTMDKMemory:
    """Initialize or load RTMDK memory."""
    config = RTMDKConfig(
        embedding_dim=768,
        latent_dim=64,
        tension_threshold=0.15,
        decay_rate=0.997,
        top_k=5,
        enable_async=False,
        learn_projection=True,
        projection_lr=0.005,
        soft_gates=True,
        self_supervision=True,
        context_format=ContextFormat.JSON,
        causal_topological=True,
        do_calculus_validation=True,
        counterfactual_enabled=True,
        meta_adaptive=True,
        self_healing=True,
        memory_tiers={"episodic", "semantic", "procedural"},
        hyperbolic=False,
        predictive_coding=True,
        counterfactual_imagination=True,
        differential_privacy=False,
        cross_modal=True,
    )

    # Try to load existing memory
    if os.path.exists(MEMORY_FILE):
        try:
            mem = RTMDKMemory.import_field(MEMORY_FILE, get_embedding)
            logger.info(f"Loaded memory from {MEMORY_FILE}: {len(mem.field.nodes)} nodes")
            return mem
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

    mem = RTMDKMemory(config=config, embedder=get_embedding)
    logger.info("Initialized new RTMDK memory")
    return mem


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

    system_prompt = "You are a helpful assistant with long-term memory powered by RTMDK (Resonance-Topological Memory)."

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
    """List available models."""
    models = [
        {
            "id": "rtmdk",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rtmdk",
            "description": "RTMDK Memory with LLM integration",
        },
        {
            "id": "rtmdk-embed",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rtmdk",
            "description": "RTMDK Embedding model",
        },
    ]
    if lm_studio_available and chat_model:
        models.append({
            "id": chat_model,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "lm-studio",
        })
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """Chat completions with RTMDK memory context."""
    if not lm_studio_available:
        raise HTTPException(
            status_code=503,
            detail="LM Studio not available. Start LM Studio and enable server on port 12345."
        )

    import requests

    # Build system prompt with memory context
    system_prompt = build_system_prompt(req.messages, req.session_id)

    # Build messages for LM Studio
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
                    {"output": ""}  # Will be updated after response
                )
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

    # Call LM Studio
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": chat_model or "local-model",
                "messages": messages,
                "temperature": req.temperature,
                "max_tokens": req.max_tokens,
                "stream": req.stream,
                "top_p": req.top_p,
                "frequency_penalty": req.frequency_penalty,
                "presence_penalty": req.presence_penalty,
            },
            timeout=120,
            stream=req.stream,
        )

        if req.stream:
            def stream_generator():
                for chunk in resp.iter_lines():
                    if chunk:
                        line = chunk.decode("utf-8")
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                # Save final response to memory
                if memory:
                    try:
                        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
                        if last_user:
                            memory.save_context(
                                {"input": last_user, "session_id": req.session_id},
                                {"output": "[streamed response]"}
                            )
                    except:
                        pass

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        data = resp.json()
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

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"LM Studio request failed: {str(e)}")


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
    if not memory:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    ctx = memory.load_memory_variables({"input": req.query, "session_id": req.session_id})
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
    """Import memory state."""
    global memory
    if not os.path.exists(MEMORY_FILE):
        raise HTTPException(status_code=404, detail=f"Memory file not found: {MEMORY_FILE}")
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

    logger.info(f"Server ready on {SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Memory nodes: {len(memory.field.nodes)}")


@app.on_event("shutdown")
async def shutdown():
    """Save memory on shutdown."""
    if memory:
        try:
            os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
            memory.export_field(MEMORY_FILE)
            logger.info(f"Memory saved to {MEMORY_FILE}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")


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
    print("    Cursor/Continue/Aider: set base URL to http://localhost:80801/v1")
    print("    API Key: rtmdk-local")
    print("-" * 60)

    uvicorn.run(
        "rtmdk_server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
