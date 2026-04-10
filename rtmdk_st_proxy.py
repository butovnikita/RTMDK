"""
rtmdk_st_proxy.py — SillyTavern Proxy Server with RTMDK Memory Integration.

This proxy sits between SillyTavern and LM Studio, providing:
- Automatic memory saving (user input + AI response)
- Memory retrieval for context injection
- Character isolation (each character has separate memory)
- Configurable RP behavior

Usage:
    python rtmdk_st_proxy.py [--port 5000] [--rtmdk http://127.0.0.1:8080] [--lm-studio http://127.0.0.1:12345]
"""

import os
import sys
import json
import time
import logging
import requests
from typing import Dict, List, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("rtmdk_st_proxy")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "rtmdk_url": "http://127.0.0.1:8080",
    "lm_studio_url": "http://127.0.0.1:12345/v1",
    "proxy_port": 5000,
    "memory": {
        "save_user_messages": True,
        "save_ai_messages": True,
        "session_per_character": True,  # Each character has separate memory
        "default_session_id": "default"
    },
    "retrieval": {
        "enabled": True,
        "max_memories": 3,  # How many memories to inject
        "insert_in_system_prompt": True,
        "memory_format": "narrative"  # "narrative" or "bullet_points"
    },
    "logging": {
        "log_memory_saves": True,
        "log_memory_retrieval": True
    }
}

class ProxyConfig:
    def __init__(self, config_path: str = "st_config.json"):
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        self._load_config()
    
    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                # Deep merge
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in self.config:
                        self.config[key].update(value)
                    else:
                        self.config[key] = value
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        else:
            self._save_config()
            logger.info(f"Created default config at {self.config_path}")
    
    def _save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @property
    def rtmdk_url(self) -> str:
        return self.config.get("rtmdk_url", "http://127.0.0.1:8080")
    
    @property
    def lm_studio_url(self) -> str:
        return self.config.get("lm_studio_url", "http://127.0.0.1:12345/v1")
    
    @property
    def proxy_port(self) -> int:
        return self.config.get("proxy_port", 5000)


# Global config
config = ProxyConfig()

# ============================================================================
# MEMORY INTEGRATION
# ============================================================================

class MemoryManager:
    """Handles all RTMDK memory operations."""
    
    def __init__(self, rtmdk_url: str):
        self.rtmdk_url = rtmdk_url
    
    def save_message(self, session_id: str, role: str, content: str) -> bool:
        """Save a message to RTMDK memory."""
        if not content.strip():
            return False
        
        try:
            # Use input/output format for RTMDK
            if role == "user":
                data = {"input": content, "session_id": session_id, "output": content}
            else:
                data = {"input": f"[{role} response]", "session_id": session_id, "output": content}
            
            resp = requests.post(
                f"{self.rtmdk_url}/v1/memory/save",
                json=data,
                timeout=10
            )
            
            if resp.ok:
                if config.config.get("logging", {}).get("log_memory_saves", True):
                    logger.info(f"Saved {role} message to memory (session: {session_id})")
                return True
            else:
                logger.warning(f"Failed to save message: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"Memory save error: {e}")
            return False
    
    def retrieve_memories(self, session_id: str, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant memories from RTMDK."""
        try:
            resp = requests.post(
                f"{self.rtmdk_url}/v1/memory/query",
                json={"query": query, "session_id": session_id, "top_k": top_k},
                timeout=10
            )
            
            if resp.ok:
                data = resp.json()
                context = data.get("rtmdk_context", "")
                if context and context not in ("No relevant memory.", "[]", ""):
                    if config.config.get("logging", {}).get("log_memory_retrieval", True):
                        logger.info(f"Retrieved memories for query: {query[:50]}...")
                    return [context]
            return []
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []
    
    def get_health(self) -> Dict:
        """Check RTMDK server health."""
        try:
            resp = requests.get(f"{self.rtmdk_url}/health", timeout=5)
            if resp.ok:
                return resp.json()
            return {"status": "error", "code": resp.status_code}
        except Exception as e:
            return {"status": "error", "message": str(e)}


# Global memory manager
memory_mgr = MemoryManager(config.rtmdk_url)

# ============================================================================
# SILLYTAVERN PROXY ENDPOINTS
# ============================================================================

app = FastAPI(title="RTMDK SillyTavern Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_session_id(request_data: Dict) -> str:
    """Extract session ID from SillyTavern request."""
    mem_config = config.config.get("memory", {})
    
    # If session per character is enabled, try to extract character name
    if mem_config.get("session_per_character", True):
        # Check for character name in various places
        char_name = (
            request_data.get("char_name") or
            request_data.get("character_name") or
            request_data.get("name") or
            "default"
        )
        return char_name
    
    return mem_config.get("default_session_id", "default")


def inject_memories_into_prompt(messages: List[Dict], memories: List[str]) -> List[Dict]:
    """Inject retrieved memories into the conversation."""
    if not memories:
        return messages
    
    mem_config = config.config.get("memory", {})
    ret_config = config.config.get("retrieval", {})
    
    # Format memories
    memory_format = ret_config.get("memory_format", "narrative")
    
    if memory_format == "bullet_points":
        memory_text = "\n".join(f"- {m}" for m in memories)
    else:
        memory_text = "\n".join(memories)
    
    memory_block = (
        f"\n\n[Relevant memories from past conversations]\n{memory_text}\n"
        f"Use these memories to inform your response and maintain continuity."
    )
    
    # Inject into system message or as first user message
    new_messages = messages.copy()
    
    # Find or create system message
    system_idx = None
    for i, msg in enumerate(new_messages):
        if msg.get("role") == "system":
            system_idx = i
            break
    
    if system_idx is not None:
        new_messages[system_idx]["content"] += memory_block
    else:
        # Prepend as system-like message
        new_messages.insert(0, {
            "role": "system",
            "content": memory_block
        })
    
    return new_messages


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """Proxy chat completions with memory integration."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    model = body.get("model", "")
    
    # Extract session ID (character name)
    session_id = extract_session_id(body)
    
    # Step 1: Save user message to memory
    mem_config = config.config.get("memory", {})
    if mem_config.get("save_user_messages", True):
        # Find last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                memory_mgr.save_message(session_id, "user", msg.get("content", ""))
                break
    
    # Step 2: Retrieve relevant memories
    ret_config = config.config.get("retrieval", {})
    memories = []
    if ret_config.get("enabled", True):
        # Use last user message as query
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break
        
        if last_user_msg:
            max_memories = ret_config.get("max_memories", 3)
            memories = memory_mgr.retrieve_memories(session_id, last_user_msg, top_k=max_memories)
    
    # Step 3: Inject memories into prompt
    if memories:
        messages = inject_memories_into_prompt(messages, memories)
    
    # Step 4: Forward to LM Studio
    lm_request = {
        "model": model,
        "messages": messages,
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_tokens", 1024),
        "stream": stream,
        "top_p": body.get("top_p", 1.0),
        "frequency_penalty": body.get("frequency_penalty", 0.0),
        "presence_penalty": body.get("presence_penalty", 0.0),
    }
    
    lm_url = f"{config.lm_studio_url}/chat/completions"
    
    if stream:
        # Handle streaming
        def stream_generator():
            try:
                with requests.post(
                    lm_url,
                    json=lm_request,
                    stream=True,
                    timeout=120
                ) as resp:
                    full_content = ""
                    for line in resp.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            yield f"{line_str}\n\n"
                            # Accumulate content for memory saving
                            if "content" in line_str:
                                import re
                                match = re.search(r'"content":"([^"]*)"', line_str)
                                if match:
                                    full_content += match.group(1)
                    
                    # Save AI response to memory after stream ends
                    if mem_config.get("save_ai_messages", True) and full_content:
                        memory_mgr.save_message(session_id, "assistant", full_content)
                        
            except Exception as e:
                logger.error(f"Streaming error: {e}")
        
        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    
    else:
        # Handle non-streaming
        try:
            resp = requests.post(lm_url, json=lm_request, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            
            # Save AI response to memory
            if mem_config.get("save_ai_messages", True):
                choices = result.get("choices", [])
                if choices:
                    ai_message = choices[0].get("message", {}).get("content", "")
                    if ai_message:
                        memory_mgr.save_message(session_id, "assistant", ai_message)
            
            return JSONResponse(content=result)
        except Exception as e:
            logger.error(f"LM Studio request failed: {e}")
            raise HTTPException(status_code=502, detail=f"LM Studio error: {str(e)}")


@app.post("/v1/completions")
async def proxy_completions(request: Request):
    """Proxy text completions for older SillyTavern API."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    # Convert to chat format
    prompt = body.get("prompt", "")
    session_id = extract_session_id(body)
    
    # Save user message
    mem_config = config.config.get("memory", {})
    if mem_config.get("save_user_messages", True) and prompt:
        memory_mgr.save_message(session_id, "user", prompt)
    
    # Retrieve memories
    ret_config = config.config.get("retrieval", {})
    memories = []
    if ret_config.get("enabled", True) and prompt:
        max_memories = ret_config.get("max_memories", 3)
        memories = memory_mgr.retrieve_memories(session_id, prompt, top_k=max_memories)
    
    # Build messages with memory
    messages = [{"role": "user", "content": prompt}]
    if memories:
        messages = inject_memories_into_prompt(messages, memories)
    
    # Forward to LM Studio
    lm_request = {
        "model": body.get("model", ""),
        "messages": messages,
        "temperature": body.get("temperature", 0.7),
        "max_tokens": body.get("max_new_tokens", body.get("max_tokens", 256)),
        "stream": body.get("stream", False),
    }
    
    lm_url = f"{config.lm_studio_url}/chat/completions"
    
    try:
        resp = requests.post(lm_url, json=lm_request, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        
        # Save AI response
        if mem_config.get("save_ai_messages", True):
            choices = result.get("choices", [])
            if choices:
                ai_message = choices[0].get("message", {}).get("content", "")
                if ai_message:
                    memory_mgr.save_message(session_id, "assistant", ai_message)
        
        # Convert back to text completion format
        text = ""
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        
        return JSONResponse(content={
            "id": "cmpl-st-proxy",
            "object": "text_completion",
            "created": int(time.time()),
            "model": body.get("model", ""),
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}]
        })
    except Exception as e:
        logger.error(f"LM Studio request failed: {e}")
        raise HTTPException(status_code=502, detail=f"LM Studio error: {str(e)}")


@app.get("/status")
async def status():
    """Proxy status with RTMDK health."""
    rtmdk_health = memory_mgr.get_health()
    return {
        "proxy": "running",
        "rtmdk": rtmdk_health,
        "config": {
            "rtmdk_url": config.rtmdk_url,
            "lm_studio_url": config.lm_studio_url,
            "memory_enabled": config.config.get("memory", {}).get("save_user_messages", True),
            "retrieval_enabled": config.config.get("retrieval", {}).get("enabled", True),
        }
    }


@app.get("/memory/stats")
async def memory_stats():
    """Get memory statistics."""
    return memory_mgr.get_health()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RTMDK SillyTavern Proxy")
    parser.add_argument("--port", type=int, default=config.proxy_port, help="Proxy port")
    parser.add_argument("--rtmdk", default=config.rtmdk_url, help="RTMDK server URL")
    parser.add_argument("--lm_studio", default=config.lm_studio_url, help="LM Studio URL")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  RTMDK SillyTavern Proxy v1.0.0")
    print("=" * 60)
    print(f"  Proxy:     http://0.0.0.0:{args.port}")
    print(f"  RTMDK:     {args.rtmdk}")
    print(f"  LM Studio: {args.lm_studio}")
    print()
    print("  SillyTavern Configuration:")
    print(f"    API Type: OpenAI")
    print(f"    Base URL: http://127.0.0.1:{args.port}/v1")
    print(f"    API Key:  (any value)")
    print("-" * 60)
    
    # Update config with CLI args
    config.config["rtmdk_url"] = args.rtmdk
    config.config["lm_studio_url"] = args.lm_studio
    config.config["proxy_port"] = args.port
    memory_mgr.rtmdk_url = args.rtmdk
    
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
