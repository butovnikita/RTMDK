# SillyTavern Integration

RTMDK gives SillyTavern characters **persistent long-term memory**: messages
are saved automatically, relevant memories are retrieved and injected into the
prompt, and each character gets isolated memory.

!!! info "Legacy scripts"
    The SillyTavern dev scripts were moved to `legacy/` during the 2026-08
    cleanup and are **frozen**. The production server is `python -m rtmdk`
    (`rtmdk/server/app.py`).

## Option 1: Monolith (simplest)

```
SillyTavern → http://127.0.0.1:8080/v1 → API Key: rtmdk-local
```

1. SillyTavern → API Connections → **API Type**: `OpenAI`
2. **Base URL**: `http://127.0.0.1:8080/v1`
3. **API Key**: `rtmdk-local`

## Option 2: Proxy (recommended)

```
SillyTavern → Proxy (5000) → RTMDK Server (8080) → LM Studio (12345)
```

```bash
python legacy/rtmdk_sillytavern_launcher.py   # starts server + proxy
```

Then point SillyTavern at `http://127.0.0.1:5000/v1` (any API key — the proxy
doesn't check it). The proxy adds automatic memory save, retrieval, and
prompt injection.

**Streaming fix (v8.3.0):** AI responses are saved to memory even with
streaming enabled.

## Full Documentation

- [SillyTavern Connection Guide — tested endpoints, both variants](../../SILLYTAVERN_CONNECTION_GUIDE.md)
- [ST Proxy Setup — architecture and step-by-step](../ST_PROXY_SETUP.md)
- [Legacy directory README — frozen scripts inventory](../../legacy/README.md)
