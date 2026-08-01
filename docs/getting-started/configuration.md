# Configuration

RTMDK is configured through a single `RTMDKConfig` object with **9 built-in presets**, overridable by **59 `RTMDK_*` environment variables**. Priority is always: explicit constructor arguments > env vars > preset defaults.

## Presets

```python
from rtmdk import RTMDKConfig, list_presets

print(list_presets())
# ['local', 'production', 'research', 'enterprise',
#  'agent', 'legal', 'medical', 'streaming', 'sillytavern']

config = RTMDKConfig.local()       # personal assistant, ~16 MB RAM, ~5 ms
config = RTMDKConfig.production()  # production server
config = RTMDKConfig.research()    # maximum accuracy
```

## Environment Variable Overrides

Any parameter can be overridden via `RTMDK_*` env vars:

```bash
RTMDK_PRESET=local RTMDK_LATENT_DIM=128 RTMDK_TOP_K=10 python -m rtmdk
```

!!! note "`.env` autoload (since 2026-08)"
    Entry points `python -m rtmdk` and `start_production.py` automatically load
    `.env` from the current directory via `python-dotenv` (real env vars win).
    A bare library import of `rtmdk.server.app` does **not** load `.env` — this
    protects test environments from side effects.

## YAML Configs

Ready-made YAML profiles live in `configs/` (`local.yaml`, `prod.yaml`,
`research.yaml`). Example (`configs/local.yaml`):

```yaml
embedding_dim: 768
latent_dim: 256
decay_rate: 0.999
max_nodes: 10000
top_k: 5
use_hnsw: true
enable_engrams: true
```

## Validation

`RTMDKConfig.validate()` returns a list of warning strings for conflicting
settings and is auto-called on initialization.

## Full Documentation

- [Fine-Tuning Guide (all 59 variables, runtime API, troubleshooting)](../05_FINE_TUNING.md)
- [API Reference §4 — full config reference](../01_API_REFERENCE.md)
