# MCP Server (Model Context Protocol)

RTMDK includes an MCP server (`rtmdk/mcp_server.py`) that exposes memory
operations as MCP tools — usable from Claude Desktop and other MCP clients.
Supports stdio and SSE transports.

## Running

```bash
python -m rtmdk.mcp                            # stdio mode (Claude Desktop)
python -m rtmdk.mcp --transport sse --port 8080
rtmdk-mcp                                      # CLI entry point
```

## Claude Desktop Config

```json
{
  "mcpServers": {
    "rtmdk": {
      "command": "python",
      "args": ["-m", "rtmdk.mcp"]
    }
  }
}
```

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `RTMDK_MCP_EMBEDDER` | `local` (sentence-transformers) or `openai` | `local` |
| `RTMDK_OPENAI_API_KEY` | Required when embedder is `openai` | — |
| `RTMDK_MCP_WAL_PATH` | WAL file path (persistence) | optional |
| `RTMDK_MCP_SNAPSHOT` | Snapshot file path | optional |
| `RTMDK_MCP_LATENT_DIM` | Embedding dimension | `384` |

If `sentence-transformers` is not installed, the server falls back to a
deterministic mock embedder and logs a warning.

## Full Documentation

- Server source & tool list: `rtmdk/mcp_server.py`
- Entry point: `rtmdk-mcp = rtmdk.mcp_server:main` (see `pyproject.toml`)
