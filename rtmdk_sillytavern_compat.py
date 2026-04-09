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
                async def stream_generator():
                    accumulated_text = ""
                    chunk_count = 0
                    last_content_time = time.time()
                    stream_timeout = 30  # seconds without data = end stream
                    
                    print(f"!!! ST STREAMING STARTED")

                    try:
                        for line in resp.iter_lines(chunk_size=64, decode_unicode=False):
                            if not line:
                                # Check for timeout on empty lines
                                if time.time() - last_content_time > stream_timeout and accumulated_text:
                                    print(f"!!! ST STREAM TIMEOUT after {stream_timeout}s")
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
                                
                                if chunk_count <= 3 or chunk_count % 10 == 0:
                                    print(f"!!! ST STREAM CHUNK {chunk_count}: {data_str[:120]}")
                                
                                if data_str.strip() == '[DONE]':
                                    print(f"!!! ST STREAM DONE marker received")
                                    # Send final result with finish_reason
                                    yield f'data: {json.dumps({"results": [{"text": accumulated_text}], "finish_reason": "stop"})}\n\n'
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    choices = chunk.get('choices', [{}])
                                    if not choices:
                                        # Empty choices = end of stream
                                        print(f"!!! ST STREAM empty choices, ending")
                                        if accumulated_text:
                                            yield f'data: {json.dumps({"results": [{"text": accumulated_text}], "finish_reason": "stop"})}\n\n'
                                        break
                                        
                                    choice = choices[0]
                                    # Check for finish_reason in the choice
                                    finish = choice.get('finish_reason')
                                    delta = choice.get('delta', {})
                                    content = delta.get('content', '')
                                    
                                    if content:
                                        accumulated_text += content
                                        last_content_time = time.time()
                                        # Silly Tavern format: {"results": [{"text": "..."}]}
                                        yield f"data: {json.dumps({'results': [{'text': accumulated_text}]})}\n\n"
                                    
                                    if finish == 'stop':
                                        # Send final with finish_reason
                                        print(f"!!! ST STREAM finish_reason=stop: {chunk_count} chunks, {len(accumulated_text)} chars")
                                        yield f'data: {json.dumps({"results": [{"text": accumulated_text}], "finish_reason": "stop"})}\n\n'
                                        break
                                        
                                    # Check if delta is empty and no finish_reason - might be end
                                    if not content and not finish and not delta:
                                        # Stream might be ending
                                        print(f"!!! ST STREAM empty delta, might be ending")
                                        if accumulated_text:
                                            yield f'data: {json.dumps({"results": [{"text": accumulated_text}], "finish_reason": "stop"})}\n\n'
                                            
                                except json.JSONDecodeError:
                                    pass
                    except Exception as e:
                        print(f"!!! ST STREAM ERROR after {chunk_count} chunks: {e}")
                    finally:
                        # Always send final result with finish_reason
                        if accumulated_text:
                            print(f"!!! ST STREAM FINAL: {len(accumulated_text)} chars, {chunk_count} chunks")
                            yield f'data: {json.dumps({"results": [{"text": accumulated_text}], "finish_reason": "stop"})}\n\n'
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
        
        # If still no stream but SillyTavern sent it, check the raw headers/body
        print(f"!!! ST REQUEST KEYS: {list(data.keys())}")
        print(f"!!! ST REQUEST stream value: {repr(data.get('stream', 'MISSING'))}")
        print(f"!!! ST REQUEST full body (first 500 chars): {json.dumps(data, ensure_ascii=False)[:500]}")
        
        return await _handle_generate(data, stream)

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
        
        print(f"!!! ST-BACKEND REQUEST KEYS: {list(data.keys())}")
        print(f"!!! ST-BACKEND stream: {repr(data.get('stream', 'MISSING'))}")
        
        return await _handle_generate(data, stream)
    
    # OpenAI completions format (for backward compatibility)
    @router.post("/v1/completions")
    async def openai_completions(request: Request):
        data = await request.json()
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)
        
        print(f"!!! OPENAI-COMPLETE KEYS: {list(data.keys())}")
        print(f"!!! OPENAI-COMPLETE stream: {repr(data.get('stream', 'MISSING'))}")
        
        result = await _handle_generate({"prompt": prompt, **data}, stream)
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": _get_chat_model() or "rtmdk",
            "choices": [{"text": r["text"], "index": i, "finish_reason": "stop"} 
                       for i, r in enumerate(result.get("results", []))],
        }
    
    return router
