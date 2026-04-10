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


def create_sillytavern_router(memory, config: Dict[str, Any], lm_studio_available_fn,
                              chat_model_fn, lm_studio_url: str, *args, **kwargs) -> APIRouter:
    """Create Silly Tavern compatible endpoints.
    
    Args:
        lm_studio_available_fn: Callable that returns current LM Studio availability
        chat_model_fn: Callable that returns current chat model name
    """
    router = APIRouter()

    def _get_mem():
        if callable(memory): return memory()
        return memory
    
    def _lm_studio_available():
        if callable(lm_studio_available_fn): return lm_studio_available_fn()
        return lm_studio_available_fn
    
    def _get_chat_model():
        if callable(chat_model_fn): return chat_model_fn()
        return chat_model_fn

    async def _handle_generate(data: dict, stream: bool = False):
        """Handle text completion request."""
        import requests

        mem = _get_mem()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_new_tokens", data.get("max_tokens", 512))
        temperature = data.get("temperature", 0.7)

        print(f"!!! ST REQUEST: prompt_len={len(prompt)}, stream={stream}")

        if not _lm_studio_available():
            print("!!! ST: LM Studio NOT available")
            return {"results": [{"text": "[Error: LLM backend not available. Start LM Studio and ensure it's running on the configured port.]"}]}

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
                    "model": _get_chat_model() or "local-model",
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
                # ST expects: {"choices":[{"text":"delta text","index":0}]} or {"content":"delta"}
                # Each chunk should contain ONLY the new delta, not cumulative text
                async def stream_generator():
                    chunk_count = 0
                    last_content_time = time.time()
                    stream_timeout = 30  # seconds without data = end stream
                    
                    print(f"!!! ST STREAMING STARTED (fixed format)")

                    try:
                        for line in resp.iter_lines(chunk_size=64, decode_unicode=False):
                            if not line:
                                if time.time() - last_content_time > stream_timeout:
                                    print(f"!!! ST STREAM TIMEOUT")
                                    break
                                continue
                            try:
                                line_str = line.decode('utf-8')
                            except UnicodeDecodeError:
                                continue
                                
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]
                                last_content_time = time.time()
                                chunk_count += 1
                                
                                if data_str.strip() == '[DONE]':
                                    print(f"!!! ST STREAM DONE marker")
                                    break
                                
                                if chunk_count <= 3 or chunk_count % 10 == 0:
                                    print(f"!!! ST STREAM CHUNK {chunk_count}: {data_str[:120]}")
                                
                                try:
                                    chunk = json.loads(data_str)
                                    choices = chunk.get('choices', [{}])
                                    
                                    if not choices:
                                        print(f"!!! ST STREAM empty choices, ending")
                                        break
                                        
                                    choice = choices[0]
                                    finish = choice.get('finish_reason')
                                    
                                    # Try different delta content locations
                                    # LM Studio chat: choices[0].delta.content
                                    # LM Studio completion: choices[0].text
                                    delta = choice.get('delta', {})
                                    content = delta.get('content', '') or choice.get('text', '')
                                    
                                    if content:
                                        # Send in ST-expected format: {"choices":[{"text":"delta","index":0}]}
                                        st_chunk = {"choices": [{"text": content, "index": 0}]}
                                        yield f"data: {json.dumps(st_chunk)}\n\n"
                                        
                                        # Also try llama.cpp format as fallback: {"content":"delta"}
                                        # ST checks both formats: data?.choices?.[0]?.text || data?.content
                                        
                                    if finish == 'stop':
                                        print(f"!!! ST STREAM finish_reason=stop")
                                        break
                                        
                                except json.JSONDecodeError:
                                    pass
                    except Exception as e:
                        print(f"!!! ST STREAM ERROR: {e}")
                    finally:
                        print(f"!!! ST STREAM ENDED: {chunk_count} chunks")
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
        
        # Debug: print full request body to see what ST actually sends
        stream = data.get("stream", False)
        
        # Check alternative stream parameter names that SillyTavern might use
        if not stream:
            for key in ["streaming", "is_streaming", "stream_mode"]:
                if data.get(key):
                    stream = True
                    print(f"!!! ST: Found stream parameter as '{key}'")
                    break
        
        print(f"!!! ST REQUEST KEYS: {list(data.keys())}")
        print(f"!!! ST REQUEST stream value: {repr(data.get('stream', 'MISSING'))}")
        
        # If streaming, return StreamingResponse directly
        if stream:
            return await _handle_generate(data, True)
        else:
            result = await _handle_generate(data, False)
            return result

    # Alternative ST endpoint
    @router.post("/api/backends/text-completions/generate")
    async def st_generate_backend(request: Request):
        data = await request.json()
        
        stream = data.get("stream", False)
        if not stream:
            for key in ["streaming", "is_streaming"]:
                if data.get(key):
                    stream = True
                    break
        
        if stream:
            return await _handle_generate(data, True)
        else:
            return await _handle_generate(data, False)
    
    @router.post("/api/backends/text-completions/status")
    async def st_status():
        """SillyTavern status check endpoint."""
        return {
            "result": "success",
            "model": _get_chat_model() or "rtmdk",
            "max_length": 4096,
            "max_context_length": 8192,
        }
    
    # OpenAI completions format (for backward compatibility)
    @router.post("/v1/completions")
    async def openai_completions(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)
        
        print(f"!!! OPENAI-COMPLETE KEYS: {list(data.keys())}")
        print(f"!!! OPENAI-COMPLETE stream: {repr(data.get('stream', 'MISSING'))}")
        
        if stream:
            return await _handle_generate({"prompt": prompt, **data}, True)
        else:
            result = await _handle_generate({"prompt": prompt, **data}, False)
            return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": _get_chat_model() or "rtmdk",
            "choices": [{"text": r["text"], "index": i, "finish_reason": "stop"}
                       for i, r in enumerate(result.get("results", []))],
        }
    
    return router
