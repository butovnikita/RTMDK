"""rtmdk_server_ux.py — FastAPI UX Router.

Adds 17 REST endpoints for UX features.

Usage:
    from rtmdk_server_ux import create_ux_router
    app.include_router(create_ux_router(memory, config))
"""
import json, time
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, PlainTextResponse

def create_ux_router(memory, config: Dict[str, Any]) -> APIRouter:
    """Create FastAPI router with all UX endpoints.
    
    memory can be:
    - RTMDKMemory instance (direct)
    - Callable that returns current memory instance
    """
    router = APIRouter(prefix="/v1", tags=["ux"])
    _m = {}

    def _get_mem():
        """Resolve current memory instance."""
        if callable(memory):
            return memory()
        return memory

    def _init():
        if _m: return
        from rtmdk.production.context_optimizer import ContextOptimizer
        from rtmdk.production.feedback_loop import FeedbackLoop
        from rtmdk.production.session_persistence import SessionPersistence
        from rtmdk.production.smart_pruning import SmartPruner
        from rtmdk.production.backup_restore import BackupManager
        from rtmdk.production.import_pipeline import ImportPipeline
        from rtmdk.production.health_monitor import HealthMonitor
        from rtmdk.production.analytics import MemoryAnalytics
        from rtmdk.production.events import EventSystem
        from rtmdk.production.tagging import TaggingSystem
        from rtmdk.production.rate_limiter import RateLimiter
        from rtmdk.production.memory_refresh import MemoryRefresh
        from rtmdk.production.export import MemoryExporter
        from rtmdk.production.embedding_cache import EmbeddingCache
        bd=config.get("RTMDK_BACKUP_DIR","/data/backups")
        sd=config.get("RTMDK_SESSION_DIR","/data/sessions")
        cd=config.get("RTMDK_CACHE_DIR","/data/embedding_cache")
        mem = _get_mem()
        if mem is None:
            raise RuntimeError("Memory not initialized yet")
        _m["co"]=ContextOptimizer(model=config.get("RTMDK_LLM_MODEL","default"),max_tokens=int(config.get("RTMDK_MAX_CONTEXT_TOKENS","300")))
        _m["fb"]=FeedbackLoop(mem,learning_rate=float(config.get("RTMDK_FEEDBACK_LR","0.05")))
        _m["ss"]=SessionPersistence(mem,save_dir=sd,auto_save_interval=int(config.get("RTMDK_AUTO_SAVE_INTERVAL","60")))
        _m["sp"]=SmartPruner(mem,max_age_days=int(config.get("RTMDK_PRUNE_AGE_DAYS","90")),min_salience=float(config.get("RTMDK_PRUNE_MIN_SALIENCE","0.05")))
        _m["bk"]=BackupManager(mem,backup_dir=bd,compression=config.get("RTMDK_BACKUP_COMPRESSION","true").lower()=="true")
        _m["ip"]=ImportPipeline(mem)
        _m["hm"]=HealthMonitor(mem)
        _m["an"]=MemoryAnalytics(mem)
        _m["ev"]=EventSystem()
        _m["tg"]=TaggingSystem(mem)
        _m["rl"]=RateLimiter(max_per_minute=int(config.get("RTMDK_RATE_LIMIT_PER_MINUTE","60")),max_per_hour=int(config.get("RTMDK_RATE_LIMIT_PER_HOUR","1000")))
        _m["mr"]=MemoryRefresh(mem)
        _m["ex"]=MemoryExporter(mem)
        _m["ec"]=EmbeddingCache(cache_dir=cd,max_size=int(config.get("RTMDK_CACHE_MAX_SIZE","100000")))
    
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
    async def bk_restore(d:dict):
        _init()
        r=_m["bk"].restore(d.get("backup_path",""))
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
    async def imp_url(d:dict):
        _init()
        if "url" not in d: raise HTTPException(400,"Missing url")
        r=_m["ip"].import_url(d["url"])
        return {"total":r.total_items,"imported":r.imported_items,"failed":r.failed_items,"errors":r.errors[:5]}
    
    @router.get("/analytics")
    async def analytics(): _init(); return _m["an"].export_report()
    
    @router.get("/health")
    async def health(): _init(); return _m["hm"].check_health()
    
    @router.get("/metrics",response_class=PlainTextResponse)
    async def metrics(): _init(); return _m["hm"].get_metrics_text()
    
    @router.get("/export")
    async def export(format:str=Query("json")):
        _init()
        if format=="markdown": return PlainTextResponse(_m["ex"].to_markdown())
        if format=="text": return PlainTextResponse(_m["ex"].to_text())
        return _m["ex"].to_dict()
    
    @router.get("/tags")
    async def tags_list(): _init(); return {"tags":_m["tg"].list_tags()}
    
    @router.get("/tags/{node_id}")
    async def tags_get(node_id:str): _init(); return {"tags":_m["tg"].get_tags_for_node(node_id)}
    
    @router.post("/tags/{node_id}")
    async def tags_add(node_id:str,d:dict):
        _init()
        for t in d.get("tags",[]): _m["tg"].add_tag(node_id,t)
        return {"added":d.get("tags",[])}
    
    @router.delete("/tags/{node_id}")
    async def tags_del(node_id:str,tag:str=Query(...)):
        _init()
        _m["tg"].remove_tag(node_id,tag)
        return {"removed":tag}
    
    @router.get("/rate-limit")
    async def rate_limit(client_id:str=Query("unknown")):
        _init()
        return {"client_id":client_id,"remaining":_m["rl"].get_remaining(client_id)}
    
    @router.get("/events")
    async def events():
        _init()
        async def gen():
            for e in _m["ev"].get_event_log(limit=50): yield f"data: {json.dumps(e)}\n\n"
            yield 'data: {"type":"end"}\n\n'
        return StreamingResponse(gen(),media_type="text/event-stream",headers={"Cache-Control":"no-cache"})
    
    @router.get("/cache/stats")
    async def cache_stats(): _init(); return _m["ec"].get_stats()

    @router.post("/cache/clear")
    async def cache_clear(): _init(); _m["ec"].clear(); return {"cleared":True}

    @router.get("/models")
    async def list_models():
        """List available LLM and embedder models from current provider."""
        import requests
        provider = config.get("RTMDK_API_PROVIDER", "lm_studio")
        api_key = config.get("OPENAI_API_KEY", "") or config.get("OPENROUTER_API_KEY", "") or config.get("ANTHROPIC_API_KEY", "")
        
        models = {"chat": [], "embedder": [], "provider": provider}
        
        try:
            if provider == "lm_studio":
                lm_url = config.get("LM_STUDIO_URL", "http://host.docker.internal:12345/v1")
                resp = requests.get(f"{lm_url}/models", timeout=10)
                if resp.ok:
                    data = resp.json()
                    models["chat"] = [m["id"] for m in data.get("data", [])]
                models["embedder"] = ["nomic-embed-text-v1.5", "all-MiniLM-L6-v2", "text-embedding-3-small"]
                
            elif provider == "openrouter":
                resp = requests.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15
                )
                if resp.ok:
                    data = resp.json()
                    all_models = data.get("data", [])
                    models["chat"] = [m["id"] for m in all_models[:50]]
                models["embedder"] = ["nomic-embed-text-v1.5", "text-embedding-3-small", "text-embedding-3-large"]
                
            elif provider == "openai":
                resp = requests.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=15
                )
                if resp.ok:
                    data = resp.json()
                    chat_models = [m["id"] for m in data.get("data", []) if "gpt" in m["id"]]
                    models["chat"] = list(set(chat_models))[:20]
                models["embedder"] = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
                
            elif provider == "anthropic":
                models["chat"] = ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-latest"]
                models["embedder"] = ["nomic-embed-text-v1.5", "text-embedding-3-small"]
                
            elif provider == "custom":
                custom_url = config.get("CUSTOM_API_URL", "")
                if custom_url:
                    resp = requests.get(f"{custom_url}/models", timeout=10)
                    if resp.ok:
                        data = resp.json()
                        models["chat"] = [m.get("id", m) for m in data.get("data", [])]
                models["embedder"] = ["nomic-embed-text-v1.5", "all-MiniLM-L6-v2"]
                
        except Exception as e:
            models["error"] = str(e)
            if not models["chat"]: models["chat"] = ["local-model"]
            if not models["embedder"]: models["embedder"] = ["nomic-embed-text-v1.5"]
        
        return models
    
    @router.post("/embedder")
    async def set_embedder(data: dict):
        """Switch embedder model at runtime."""
        model = data.get("model", "")
        if model:
            config["RTMDK_EMBED_MODEL"] = model
            return {"status": "ok", "model": model}
        return {"error": "Missing model name"}

    return router
