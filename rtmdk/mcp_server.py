"""RTMDK MCP Server — Model Context Protocol integration.

Usage:
    python -m rtmdk.mcp              # stdio mode (Claude Desktop)
    python -m rtmdk.mcp --transport sse --port 8080
    rtmdk-mcp                        # CLI entry point

Environment variables:
    RTMDK_MCP_EMBEDDER    — "local" (default, sentence-transformers) or "openai"
    RTMDK_OPENAI_API_KEY  — required when embedder=openai
    RTMDK_MCP_WAL_PATH    — WAL file path (optional)
    RTMDK_MCP_SNAPSHOT    — snapshot file path (optional)
    RTMDK_MCP_LATENT_DIM  — embedding dimension (default 384)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, Optional

import numpy as np

from rtmdk.memory.core import RTMDKMemory
from rtmdk.memory.config import RTMDKConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedder factory
# ---------------------------------------------------------------------------

def _get_embedder(dim: int = 384):
    embedder_type = os.environ.get("RTMDK_MCP_EMBEDDER", "local").lower()
    if embedder_type == "openai":
        api_key = os.environ.get("RTMDK_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI embedder selected but RTMDK_OPENAI_API_KEY not set")
        import openai
        client = openai.OpenAI(api_key=api_key)

        def _openai_embed(text: str) -> np.ndarray:
            resp = client.embeddings.create(input=text, model="text-embedding-3-small")
            return np.array(resp.data[0].embedding, dtype=np.float32)

        return _openai_embed

    # Local sentence-transformers fallback
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        def _local_embed(text: str) -> np.ndarray:
            return model.encode(text, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

        return _local_embed
    except ImportError:
        logger.warning("sentence-transformers not installed; using mock embedder")

        def _mock_embed(text: str) -> np.ndarray:
            h = hash(text) % (2 ** 32)
            rng = np.random.default_rng(h)
            return rng.standard_normal(dim, dtype=np.float32)

        return _mock_embed


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent
except ImportError as exc:
    raise ImportError("mcp package required. Install: pip install mcp") from exc


class _MemoryContext:
    """Shared context between lifespan and request handlers."""

    def __init__(self):
        self.memory: Optional[RTMDKMemory] = None


_ctx = _MemoryContext()


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Initialize RTMDK on startup, snapshot on shutdown."""
    dim = int(os.environ.get("RTMDK_MCP_LATENT_DIM", "384"))
    wal_path = os.environ.get("RTMDK_MCP_WAL_PATH")
    snapshot_path = os.environ.get("RTMDK_MCP_SNAPSHOT")

    embedder = _get_embedder(dim)
    cfg = RTMDKConfig(
        latent_dim=dim,
        use_hnsw=True,
        hyperbolic=False,
        quantization="none",
        enable_engrams=False,
    )

    if snapshot_path and os.path.exists(snapshot_path):
        logger.info(f"Loading snapshot from {snapshot_path}")
        _ctx.memory = RTMDKMemory.import_field(
            snapshot_path, embedder=embedder, config=cfg, wal_path=wal_path
        )
    else:
        _ctx.memory = RTMDKMemory(config=cfg, embedder=embedder, wal_path=wal_path)

    # Graceful shutdown handler
    def _on_sigterm(signum, frame):
        logger.info("Received SIGTERM, exporting snapshot...")
        if snapshot_path and _ctx.memory is not None:
            _ctx.memory.export_field(snapshot_path)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    logger.info("RTMDK MCP server ready")
    yield {"memory": _ctx.memory}

    # Shutdown
    if snapshot_path and _ctx.memory is not None:
        logger.info(f"Exporting snapshot to {snapshot_path}")
        _ctx.memory.export_field(snapshot_path)


mcp = FastMCP("rtmdk", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def add_memory(text: str, session_id: str = "default", modality: str = "text") -> str:
    """Add a text memory to the resonance-topological field.

    Args:
        text: The text to remember.
        session_id: Optional session/character ID.
        modality: "text", "image", "audio", etc.
    """
    if _ctx.memory is None:
        return "Error: memory not initialized"
    _ctx.memory.save_context(
        {"input": text, "session_id": session_id},
        {"output": ""},
    )
    return f"Memory added: {text[:80]}..."


@mcp.tool()
def query_memory(query: str, top_k: int = 5, session_id: str = "default") -> str:
    """Query the memory field for relevant context.

    Args:
        query: The query text.
        top_k: Number of top results to return.
        session_id: Optional session filter.
    """
    if _ctx.memory is None:
        return "Error: memory not initialized"
    embedding = _ctx.memory.embedder(query)
    results = _ctx.memory.field.query(embedding, top_k=top_k)
    lines = []
    for nid, score, node in results:
        text = node.content.get("text") or node.content.get("input_text", "")
        lines.append(f"- {text[:120]} (score={score:.3f}, id={nid})")
    return "\n".join(lines) if lines else "No relevant memories found."


@mcp.tool()
def delete_memory(node_id: str) -> str:
    """Delete a memory node by ID.

    Args:
        node_id: The node ID to delete.
    """
    if _ctx.memory is None:
        return "Error: memory not initialized"
    if node_id not in _ctx.memory.field.nodes:
        return f"Node {node_id} not found"
    _ctx.memory.field.delete_nodes([node_id])
    return f"Deleted {node_id}"


@mcp.tool()
def consolidate_memory() -> str:
    """Run memory consolidation (merge similar nodes, prune weak ones)."""
    if _ctx.memory is None:
        return "Error: memory not initialized"
    before = len(_ctx.memory.field.nodes)
    _ctx.memory.field.consolidate()
    after = len(_ctx.memory.field.nodes)
    return f"Consolidation complete: {before} → {after} nodes"


@mcp.tool()
def get_memory_stats() -> str:
    """Return memory field statistics as JSON."""
    if _ctx.memory is None:
        return "Error: memory not initialized"
    stats = dict(_ctx.memory.field.stats)
    stats["active_nodes"] = len(_ctx.memory.field.nodes)
    return json.dumps(stats, indent=2, default=str)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("memory://stats")
def memory_stats() -> str:
    """Field statistics."""
    return get_memory_stats()


@mcp.resource("memory://nodes")
def memory_nodes() -> str:
    """List of active node IDs."""
    if _ctx.memory is None:
        return "[]"
    return json.dumps(list(_ctx.memory.field.nodes.keys()))


@mcp.resource("memory://node/{node_id}")
def memory_node(node_id: str) -> str:
    """Single node content."""
    if _ctx.memory is None:
        return "{}"
    node = _ctx.memory.field.nodes.get(node_id)
    if node is None:
        return f'{{"error": "Node {node_id} not found"}}'
    return json.dumps(
        {
            "id": node.id,
            "content": node.content,
            "modality": node.modality,
            "phase": node.phase,
            "amplitude": node.amplitude,
            "salience": node.salience,
            "created_at": node.created_at,
        },
        indent=2,
        default=str,
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

@mcp.prompt("memory://prompts/context")
def memory_context_prompt(query: str = "") -> str:
    """Generate a system prompt enriched with relevant memory context."""
    if _ctx.memory is None or not query:
        return "You are a helpful assistant with long-term memory."
    embedding = _ctx.memory.embedder(query)
    results = _ctx.memory.field.query(embedding, top_k=5)
    context_parts = []
    for nid, score, node in results:
        text = node.content.get("text") or node.content.get("input_text", "")
        if text:
            context_parts.append(f"- {text[:200]}")
    context = "\n".join(context_parts)
    return (
        "You are a helpful assistant with long-term memory.\n\n"
        f"Relevant past context for the current topic:\n{context}\n\n"
        "Use the above context to inform your response when relevant."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RTMDK MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse", "streamable-http"],
        default="stdio", help="Transport protocol"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for SSE/HTTP")
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE/HTTP")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
