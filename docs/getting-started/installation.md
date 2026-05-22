# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- numpy >= 1.24
- scipy >= 1.10

## Basic Install

```bash
pip install rtmdk
```

## With Optional Dependencies

```bash
# SOT v2 tokenizer (recommended)
pip install rtmdk[sot]

# MCP server support
pip install rtmdk[mcp]

# LangChain integration
pip install rtmdk[langchain]

# LlamaIndex integration
pip install rtmdk[llamaindex]

# Development (tests, linting)
pip install rtmdk[dev]

# All extras
pip install rtmdk[sot,mcp,langchain,llamaindex,dev]
```

## Docker

```bash
docker pull rtmdk/rtmdk:latest
docker run -p 8080:8080 rtmdk/rtmdk:latest
```

## Verify Installation

```python
from rtmdk import RTMDKMemory, RTMDKConfig
print(RTMDKConfig.local())
```
