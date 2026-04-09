"""
rtmdk_sillytavern_compat.py — Silly Tavern Text Completions API compatibility.

Adds endpoints that Silly Tavern expects when using Text Completion API type:
- POST /api/v1/generate
- POST /api/backends/text-completions/generate

These forward to the same chat completion logic as /v1/chat/completions.
"""

import json
import time
import os
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("rtmdk.st_compat")


def create_sillytavern_router(memory, config: Dict[str, Any], lm_studio_available: bool,
                              chat_model: str, lm_studio_url: str, *args, **kwargs) -> APIRouter:
    """Create Silly Tavern compatible endpoints.
    
    memory can be an instance or a callable returning current instance.
    """
    router = APIRouter()

    def _get_mem():
        if callable(memory): return memory()
        return memory

    async def _handle_generate(data: dict, stream: bool = False):
        """Handle text completion request."""
        import requests

        mem = _get_mem()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_new_tokens", data.get("max_tokens", 512))
        temperature = data.get("temperature", 0.7)

        if not lm_studio_available:
            return {"results": [{"text": "[Error: LLM backend not available. Start LM Studio or configure provider.]"}]}

        # Build system prompt with memory context
        session_id = data.get("session_id", "default")
        if mem and prompt:
            try:
                mem.save_context(
                    {"input": prompt, "session_id": session_id},
                    {"output": ""}
                )
            except Exception as e:
                logger.warning(f"Memory save failed: {e}")

        messages = []
        if mem:
            ctx = mem.load_memory_variables({"input": prompt, "session_id": session_id})
            context = ctx.get("rtmdk_context", "")
            if context:
                messages.append({"role": "system", "content": f"Use this context: {context}"})
        
        messages.append({"role": "user", "content": prompt})
        
        lm_timeout = int(os.getenv("RTMDK_LM_STUDIO_TIMEOUT", "120"))
        
        try:
            resp = requests.post(
                f"{lm_studio_url}/chat/completions",
                json={
                    "model": chat_model or "local-model",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": stream,
                },
                timeout=lm_timeout,
                stream=stream,
            )
            
            if stream:
                # Convert LM Studio streaming to Silly Tavern format
                async def stream_generator():
                    accumulated_text = ""
                    try:
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                if data_str.strip() == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        accumulated_text += content
                                        # Silly Tavern format: {"results": [{"text": "..."}]}
                                        yield f"data: {json.dumps({'results': [{'text': accumulated_text}]})}\n\n"
                                except json.JSONDecodeError:
                                    pass
                    except Exception as e:
                        logger.error(f"Stream error: {e}")
                    finally:
                        # Send final result
                        yield f"data: {json.dumps({'results': [{'text': accumulated_text}]})}\n\n"
                        yield 'data: [DONE]\n\n'
                
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            else:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {"results": [{"text": text}]}
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return {"results": [{"text": f"[Error: LLM request failed: {e}]"}]}
    
    # Text Generation WebUI format (most common for ST)
    @router.post("/api/v1/generate")
    async def st_generate_v1(request: Request):
        data = await request.json()
        return await _handle_generate(data, data.get("stream", False))

    # Alternative ST endpoint
    @router.post("/api/backends/text-completions/generate")
    async def st_generate_backend(request: Request):
        data = await request.json()
        return await _handle_generate(data, data.get("stream", False))
    
    # OpenAI completions format (for backward compatibility)
    @router.post("/v1/completions")
    async def openai_completions(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        result = await _handle_generate({"prompt": prompt, **data})
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": chat_model or "rtmdk",
            "choices": [{"text": r["text"], "index": i, "finish_reason": "stop"} 
                       for i, r in enumerate(result.get("results", []))],
        }
    
    return router
