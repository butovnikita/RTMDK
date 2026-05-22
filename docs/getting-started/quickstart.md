# Quick Start

Get RTMDK running in 5 minutes.

## Installation

```bash
pip install rtmdk
```

For SOT v2 (Self-Organizing Tokenizer):

```bash
pip install rtmdk[sot]
```

## Basic Usage

```python
from rtmdk import RTMDKMemory, RTMDKConfig
import numpy as np

cfg = RTMDKConfig.local()
memory = RTMDKMemory(
    config=cfg,
    embedder=lambda text: np.random.randn(64).astype(np.float32)
)

memory.add_node("The sky is blue")
memory.add_node("Grass is green")

results = memory.retrieve_nodes("What color is the sky?", top_k=3)
for r in results:
    print(r["text"], r["score"])
```

## Run Server

```bash
python -m rtmdk
# → http://localhost:8080
```

## Docker

```bash
docker-compose -f docker-compose.prod.yml up -d
```
