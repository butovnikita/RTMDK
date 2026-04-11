"""rtmdk_server_ux.py — FastAPI UX Router.

Adds 17 REST endpoints for UX features.

Usage:
    from rtmdk_server_ux import create_ux_router
    app.include_router(create_ux_router(memory, config))
"""
import json, time, os, logging, importlib
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from rtmdk_memory_v8 import RTMDKMemory

logger = logging.getLogger("rtmdk.ux")

def _safe_import(module_path: str, **kwargs):
    """Safely import a class from a module path, return None if it fails."""
    try:
        # Split into module and class
        parts = module_path.rsplit(".", 1)
        if len(parts) != 2:
            return None
        module_path_str, class_name = parts
        
        # Try standard import (will trigger __init__.py chain)
        mod = __import__(module_path_str, fromlist=[class_name])
        cls = getattr(mod, class_name, None)
        if cls:
            return cls(**kwargs)
    except Exception as e:
        # Log only first failure per module
        if not hasattr(_safe_import, '_logged'):
            _safe_import._logged = set()
        if module_path_str not in _safe_import._logged:
            _safe_import._logged.add(module_path_str)
            logger.debug(f"Module {module_path_str} unavailable: {type(e).__name__}")
    return None

def create_ux_router(memory, config: Dict[str, Any]) -> APIRouter:
    """Create FastAPI router with all UX endpoints.

    memory can be:
    - RTMDKMemory instance (direct)
    - Callable that returns current memory instance
    """
    router = APIRouter(prefix="/api", tags=["ux"])
    _m = {}
    _initialized = False

    def _get_mem():
        """Resolve current memory instance."""
        if callable(memory):
            return memory()
        return memory

    def _init():
        """Initialize all UX modules once, with error handling for each."""
        nonlocal _initialized
        if _initialized:
            return

        mem = _get_mem()
        if mem is None:
            logger.warning("Memory not initialized yet, UX modules deferred")
            return

        # Directories
        bd = config.get("RTMDK_BACKUP_DIR", os.path.join(os.path.expanduser("~"), ".rtmdk", "backups"))
        sd = config.get("RTMDK_SESSION_DIR", os.path.join(os.path.expanduser("~"), ".rtmdk", "sessions"))
        cd = config.get("RTMDK_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".rtmdk", "embedding_cache"))

        # Create directories if they don't exist
        for d in [bd, sd, cd]:
            try:
                os.makedirs(d, exist_ok=True)
            except:
                pass

        # Initialize each module independently with error handling
        _m["co"] = _safe_import("rtmdk.production.context_optimizer.ContextOptimizer",
            model=config.get("RTMDK_LLM_MODEL", "default"),
            max_tokens=int(config.get("RTMDK_MAX_CONTEXT_TOKENS", "300")))

        _m["fb"] = _safe_import("rtmdk.production.feedback_loop.FeedbackLoop",
            memory=mem, learning_rate=float(config.get("RTMDK_FEEDBACK_LR", "0.05")))

        _m["ss"] = _safe_import("rtmdk.production.session_persistence.SessionPersistence",
            memory=mem, save_dir=sd, auto_save_interval=int(config.get("RTMDK_AUTO_SAVE_INTERVAL", "60")))

        _m["sp"] = _safe_import("rtmdk.production.smart_pruning.SmartPruner",
            memory=mem, max_age_days=int(config.get("RTMDK_PRUNE_AGE_DAYS", "90")),
            min_salience=float(config.get("RTMDK_PRUNE_MIN_SALIENCE", "0.05")))

        _m["bk"] = _safe_import("rtmdk.production.backup_restore.BackupManager",
            memory=mem, backup_dir=bd,
            compression=config.get("RTMDK_BACKUP_COMPRESSION", "true").lower() == "true")

        _m["ip"] = _safe_import("rtmdk.production.import_pipeline.ImportPipeline", memory=mem)
        _m["hm"] = _safe_import("rtmdk.production.health_monitor.HealthMonitor", memory=mem)
        _m["an"] = _safe_import("rtmdk.production.analytics.MemoryAnalytics", memory=mem)
        _m["ev"] = _safe_import("rtmdk.production.events.EventSystem")
        _m["tg"] = _safe_import("rtmdk.production.tagging.TaggingSystem", memory=mem)
        _m["rl"] = _safe_import("rtmdk.production.rate_limiter.RateLimiter",
            max_per_minute=int(config.get("RTMDK_RATE_LIMIT_PER_MINUTE", "60")),
            max_per_hour=int(config.get("RTMDK_RATE_LIMIT_PER_HOUR", "1000")))
        _m["mr"] = _safe_import("rtmdk.production.memory_refresh.MemoryRefresh", memory=mem)
        _m["ex"] = _safe_import("rtmdk.production.export.MemoryExporter", memory=mem)
        _m["ec"] = _safe_import("rtmdk.production.embedding_cache.EmbeddingCache",
            cache_dir=cd, max_size=int(config.get("RTMDK_CACHE_MAX_SIZE", "100000")))

        _initialized = True
        loaded = sum(1 for v in _m.values() if v is not None)
        logger.info(f"UX modules initialized: {loaded}/{len(_m)}")
    
    @router.post("/feedback")
    async def fb(d:dict={}):
        _init()
        if "query" not in d: raise HTTPException(400,"Missing query")
        return _m["fb"].apply_feedback(d["query"],float(d.get("quality",0.5)),d.get("session_id","default"))
    
    @router.get("/feedback/stats")
    async def fb_stats(): _init(); return _m["fb"].get_stats()
    
    @router.post("/session/save")
    async def ss_save(d:dict={}):
        _init()
        return {"saved":True,"path":_m["ss"].save_session(d.get("session_id","default"),d.get("metadata",{}))}
    
    @router.post("/session/load")
    async def ss_load(d:dict={}):
        _init()
        r=_m["ss"].load_session(d.get("session_id","default"))
        if r is None: raise HTTPException(404,"Session not found")
        return r
    
    @router.get("/session/list")
    async def ss_list(): _init(); return {"sessions":_m["ss"].list_sessions()}
    
    @router.post("/backup/create")
    async def bk_create(d:dict={}):
        _init()
        rot=config.get("RTMDK_BACKUP_ROTATION","0")
        return {"created":True,"path":_m["bk"].create_backup(d.get("name",""),auto_rotate=rot!="0",max_backups=int(rot) if rot!="0" else 5)}
    
    @router.post("/backup/restore")
    async def bk_restore(d:dict={}):
        _init()
        backup_path = d.get("backup_path","")
        # C4: Path traversal prevention — validate backup_path
        if ".." in backup_path or backup_path.startswith("/") or backup_path.startswith("\\"):
            raise HTTPException(400, "Invalid backup path")
        backup_dir = config.get("RTMDK_BACKUP_DIR", os.path.join(os.path.expanduser("~"), ".rtmdk", "backups"))
        full_path = os.path.normpath(os.path.join(backup_dir, backup_path))
        if not full_path.startswith(os.path.normpath(backup_dir)):
            raise HTTPException(400, "Backup path outside allowed directory")
        if not _m.get("bk"):
            raise HTTPException(503, "Backup module not available")
        r=_m["bk"].restore(full_path)
        if not r or not isinstance(r, dict):
            raise HTTPException(500, "Restore returned invalid response")
        if not r.get("success"): raise HTTPException(400,r.get("error","Restore failed"))
        return r
    
    @router.get("/backup/list")
    async def bk_list(): _init(); return {"backups":_m["bk"].list_backups()}
    
    @router.post("/import/json")
    async def imp_json(d:list,text_field:str=Query("text")):
        _init()
        r=_m["ip"]._import_items(d,text_field,None,None,0)
        return {"total":r.total_items,"imported":r.imported_items,"failed":r.failed_items,"duration_s":round(r.duration_seconds,1)}
    
    @router.post("/import/url")
    async def imp_url(d:dict={}):
        _init()
        if "url" not in d: raise HTTPException(400,"Missing url")
        url = d["url"]
        # C5: SSRF prevention — block internal/private IPs
        import re
        blocked_patterns = [
            r'localhost', r'127\.0\.0\.1', r'0\.0\.0\.0',
            r'10\.\d+\.\d+\.\d+', r'172\.(1[6-9]|2\d|3[01])\.\d+\.\d+',
            r'192\.168\.\d+\.\d+', r'169\.254\.\d+\.\d+',
            r'::1', r'\[::', r'file://', r'ftp://',
        ]
        for pattern in blocked_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                raise HTTPException(400, f"URL blocked for security: internal/private address not allowed")
        if not _m.get("ip"):
            raise HTTPException(503, "Import module not available")
        r=_m["ip"].import_url(url)
        return {"total":r.total_items,"imported":r.imported_items,"failed":r.failed_items,"errors":r.errors[:5]}
    
    @router.get("/analytics")
    async def analytics():
        _init()
        if not _m.get("an"): raise HTTPException(503, "Analytics module not available")
        return _m["an"].export_report()
    
    @router.get("/health")
    async def health():
        _init()
        mem = _get_mem()
        node_count = len(mem.field.nodes) if mem else 0

        result = {"status": "ok", "nodes": node_count}

        if _m.get("hm"):
            try:
                hm_result = _m["hm"].check_health()
                result.update(hm_result)
            except Exception as e:
                result["health_monitor_error"] = str(e)

        # Ensure consistent node count format
        result["node_count"] = node_count
        result["memory_nodes"] = node_count
        result["checks"] = result.get("checks", {})
        result["checks"]["node_count"] = result["checks"].get("node_count", {"value": node_count})
        result["checks"]["node_count"]["value"] = node_count
        return result
    
    @router.get("/metrics",response_class=PlainTextResponse)
    async def metrics():
        _init()
        if not _m.get("hm"): raise HTTPException(503, "Health monitor not available")
        return _m["hm"].get_metrics_text()

    @router.get("/export")
    async def export(format:str=Query("json")):
        _init()
        if not _m.get("ex"): raise HTTPException(503, "Export module not available")
        if format=="markdown": return PlainTextResponse(_m["ex"].to_markdown())
        if format=="text": return PlainTextResponse(_m["ex"].to_text())
        return _m["ex"].to_dict()

    @router.get("/tags")
    async def tags_list():
        _init()
        if not _m.get("tg"): raise HTTPException(503, "Tagging module not available")
        return {"tags":_m["tg"].list_tags()}

    @router.get("/tags/{node_id}")
    async def tags_get(node_id:str):
        _init()
        if not _m.get("tg"): raise HTTPException(503, "Tagging module not available")
        return {"tags":_m["tg"].get_tags_for_node(node_id)}

    @router.post("/tags/{node_id}")
    async def tags_add(node_id:str,d:dict={}):
        _init()
        if not _m.get("tg"): raise HTTPException(503, "Tagging module not available")
        for t in d.get("tags",[]):
            if isinstance(t, str): _m["tg"].add_tag(node_id,t)
        return {"added":d.get("tags",[])}
    
    @router.delete("/tags/{node_id}")
    async def tags_del(node_id:str,tag:str=Query(...)):
        _init()
        if not _m.get("tg"): raise HTTPException(503, "Tagging module not available")
        _m["tg"].remove_tag(node_id,tag)
        return {"removed":tag}

    @router.get("/rate-limit")
    async def rate_limit(client_id:str=Query("unknown")):
        _init()
        if not _m.get("rl"): return {"client_id": client_id, "remaining": -1, "message": "Rate limiter not available"}
        return {"client_id":client_id,"remaining":_m["rl"].get_remaining(client_id)}

    @router.get("/events")
    async def events():
        _init()
        if not _m.get("ev"):
            async def gen_no_events():
                yield 'data: {"type":"end","message":"Event log not available"}\n\n'
            return StreamingResponse(gen_no_events(),media_type="text/event-stream",headers={"Cache-Control":"no-cache"})
        async def gen():
            try:
                for e in _m["ev"].get_event_log(limit=50): yield f"data: {json.dumps(e)}\n\n"
            except Exception:
                pass
            yield 'data: {"type":"end"}\n\n'
        return StreamingResponse(gen(),media_type="text/event-stream",headers={"Cache-Control":"no-cache"})
    
    @router.get("/cache/stats")
    async def cache_stats():
        _init()
        if not _m.get("ec"):
            return {"hit_rate": 0, "total": 0, "hits": 0, "message": "EmbeddingCache not available"}
        return _m["ec"].get_stats()

    @router.post("/cache/clear")
    async def cache_clear():
        _init()
        if not _m.get("ec"):
            return {"cleared": False, "message": "EmbeddingCache not available"}
        _m["ec"].clear()
        return {"cleared": True}

    @router.get("/models")
    async def list_models_ux():
        """List models with UX-specific format (chat, embedder, provider).
        This is separate from the main /v1/models OpenAI endpoint."""
        import requests
        import asyncio
        provider = config.get("RTMDK_API_PROVIDER", "lm_studio")
        api_key = config.get("OPENAI_API_KEY", "") or config.get("OPENROUTER_API_KEY", "") or config.get("ANTHROPIC_API_KEY", "")

        models = {"chat": [], "embedder": [], "provider": provider}

        def _fetch_models():
            try:
                if provider == "lm_studio":
                    lm_url = config.get("LM_STUDIO_URL", "http://localhost:12345/v1")
                    resp = requests.get(f"{lm_url}/models", timeout=5)
                    if resp.ok:
                        data = resp.json()
                        models["chat"] = [m["id"] for m in data.get("data", [])]
                    models["embedder"] = [m["id"] for m in data.get("data", []) if "embed" in m["id"].lower()] if resp.ok else ["nomic-embed-text-v1.5"]

                elif provider == "openrouter":
                    resp = requests.get(
                        "https://openrouter.ai/api/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10
                    )
                    if resp.ok:
                        data = resp.json()
                        models["chat"] = [m["id"] for m in data.get("data", [])[:50]]
                    models["embedder"] = ["nomic-embed-text-v1.5", "text-embedding-3-small"]

                elif provider == "openai":
                    resp = requests.get(
                        "https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10
                    )
                    if resp.ok:
                        data = resp.json()
                        # H5: Include o1, o3, o4 models (not just gpt)
                        chat_models = [m["id"] for m in data.get("data", [])
                                       if any(x in m["id"] for x in ["gpt", "o1", "o3", "o4", "claude"])]
                        models["chat"] = chat_models[:20]
                    models["embedder"] = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

                elif provider == "anthropic":
                    models["chat"] = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-latest"]
                    models["embedder"] = ["nomic-embed-text-v1.5", "text-embedding-3-small"]

                elif provider == "custom":
                    custom_url = config.get("CUSTOM_API_URL", "")
                    if custom_url:
                        resp = requests.get(f"{custom_url}/models", timeout=5)
                        if resp.ok:
                            data = resp.json()
                            models["chat"] = [m.get("id", m) for m in data.get("data", [])]
                    models["embedder"] = ["nomic-embed-text-v1.5", "all-MiniLM-L6-v2"]
            except Exception as e:
                models["error"] = str(e)
                if not models["chat"]: models["chat"] = ["local-model"]
                if not models["embedder"]: models["embedder"] = ["nomic-embed-text-v1.5"]

        # H5: Run blocking requests in thread pool to avoid blocking event loop
        try:
            await asyncio.to_thread(_fetch_models)
        except Exception:
            # Fallback for Python < 3.9 or Windows issues
            _fetch_models()

        return models
    
    @router.post("/embedder")
    async def set_embedder(data: dict):
        """Switch embedder model at runtime."""
        model = data.get("model", "")
        if model:
            config["RTMDK_EMBED_MODEL"] = model
            return {"status": "ok", "model": model}
        return {"error": "Missing model name"}
    
    @router.post("/config")
    async def update_config(data: dict):
        """Update server configuration at runtime."""
        updates = {}
        
        # Supported runtime config keys
        allowed_keys = [
            "RTMDK_API_PROVIDER", "RTMDK_LLM_MODEL", "RTMDK_EMBED_MODEL",
            "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
            "CUSTOM_API_URL", "LM_STUDIO_URL", "RTMDK_PORT"
        ]
        
        for key in allowed_keys:
            if key in data:
                config[key] = data[key]
                updates[key] = data[key]
            elif key.lower() in data:
                # Support lowercase keys too
                config[key] = data[key.lower()]
                updates[key] = data[key.lower()]
        
        if updates:
            # Save to .env file for persistence
            env_path = ".env"
            try:
                existing = {}
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if '=' in line and not line.startswith('#'):
                                k, v = line.strip().split('=', 1)
                                existing[k] = v
                
                existing.update(updates)
                
                with open(env_path, 'w') as f:
                    for k, v in existing.items():
                        f.write(f"{k}={v}\n")
            except Exception as e:
                return {"status": "partial", "updates": updates, "warning": f"Failed to persist config: {e}"}
        
        return {"status": "ok", "updates": list(updates.keys())}
    
    @router.get("/config")
    async def get_config():
        """Get current runtime configuration."""
        return {
            "provider": config.get("RTMDK_API_PROVIDER", "lm_studio"),
            "llm_model": config.get("RTMDK_LLM_MODEL", "default"),
            "embed_model": config.get("RTMDK_EMBED_MODEL", "nomic-embed-text-v1.5"),
            "lm_studio_url": config.get("LM_STUDIO_URL", "http://localhost:12345/v1"),
            "custom_url": config.get("CUSTOM_API_URL", ""),
        }

    @router.post("/backup/upload")
    async def upload_backup(request: Request):
        """Upload and restore a memory backup file."""
        import tempfile

        mem = _get_mem()
        if not mem:
            raise HTTPException(503, "Memory not initialized")

        # Parse multipart form data
        form = await request.form()
        file = form.get("file")
        if not file or not hasattr(file, 'filename'):
            raise HTTPException(400, "No file provided")

        # H6: Limit upload size to 100MB
        content = await file.read()
        max_size = 100 * 1024 * 1024  # 100MB
        if len(content) > max_size:
            raise HTTPException(413, f"File too large: {len(content) / 1024 / 1024:.1f}MB (max 100MB)")

        # Save temp file safely
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                f.write(content)
                temp_path = f.name

            # Restore memory from backup
            mem2 = RTMDKMemory.import_field(temp_path, mem.embedder)
            if not mem2 or len(mem2.field.nodes) == 0:
                raise HTTPException(400, "Failed to restore: no nodes found")

            # Copy nodes to current memory
            mem.field.nodes.clear()
            mem.field.node_index.clear()
            for nid, node in mem2.field.nodes.items():
                mem.field.nodes[nid] = node
                mem.field.node_index.append(nid)

            # Copy stats
            mem.field.stats.update(mem2.field.stats)

            node_count = len(mem.field.nodes)
            return {"status": "ok", "nodes_restored": node_count}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Restore failed: {str(e)}")
        finally:
            import os
            if temp_path:
                try: os.unlink(temp_path)
                except: pass

    return router
