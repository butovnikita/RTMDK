"""
rtmdk/production/streaming.py — Streaming Response Support.

Compatible with OpenAI streaming API for token-by-token responses.
"""

from typing import Generator, Dict, Any, Iterator


class StreamingResponse:
    """Generates streaming-compatible response chunks.
    
    Usage:
        streamer = StreamingResponse(memory, llm_callback)
        for chunk in streamer.stream_response("What do I know about coffee?"):
            print(chunk["content"], end="", flush=True)
    """
    
    def __init__(self, memory, llm_callback):
        self.memory = memory
        self.llm_callback = llm_callback  # function(query, context) → response text
    
    def stream_response(
        self,
        query: str,
        session_id: str = "default",
        chunk_size: int = 5,  # tokens per chunk
    ) -> Iterator[Dict[str, Any]]:
        """Stream response token by token.
        
        Yields:
            {"content": str, "finish_reason": str or None}
        """
        # Get context from RTMDK
        ctx = self.memory.load_memory_variables({
            "input": query,
            "session_id": session_id,
        })
        context = ctx.get("rtmdk_context", "")
        
        # Get full response from LLM
        full_response = self.llm_callback(query, context)
        
        # Stream in chunks
        words = full_response.split()
        current_chunk = []
        
        for i, word in enumerate(words):
            current_chunk.append(word)
            
            if len(current_chunk) >= chunk_size or i == len(words) - 1:
                yield {
                    "content": " ".join(current_chunk) + (" " if i < len(words) - 1 else ""),
                    "finish_reason": None,
                }
                current_chunk = []
        
        yield {"content": "", "finish_reason": "stop"}
