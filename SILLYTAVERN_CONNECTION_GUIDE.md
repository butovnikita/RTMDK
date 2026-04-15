# RTMDK SillyTavern Connection Guide

## Server Status: WORKING ✅

All endpoints tested and verified:
- `/health` → Returns server health
- `/v1/models` → Returns available models
- `/v1/chat/completions` → OpenAI chat format (WORKS)
- `/v1/embeddings` → Embedding endpoint (WORKS)
- `/api/v1/generate` → SillyTavern text completion format)
- `/api/backends/text-completions/generate` → ST backend format (WORKS)
- `/api/backends/text-completions/status` → ST status check (WORKS)
- `/dashboard` → Web UI (WORKS)

**Streaming fix (v8.1.0):** AI responses are now automatically saved to memory even when streaming is enabled.

## Подключение

### Вариант 1: Monolith (проще)

```
SillyTavern → http://127.0.0.1:8080/v1 → API Key: rtmdk-local
```

1. SillyTavern → API Connections → **API Type**: `OpenAI`
2. Base URL: `http://127.0.0.1:8080/v1`
3. API Key: `rtmdk-local`
4. Connect

### Вариант 2: Proxy (рекомендуется)

```
SillyTavern → http://127.0.0.1:5000/v1 → RTMDK Server (8080) → LM Studio (12345)
```

```bash
python rtmdk_sillytavern_launcher.py
# Запускает сервер (8080) + proxy (5000)
```

1. SillyTavern → API Connections → **API Type**: `OpenAI`
2. Base URL: `http://127.0.0.1:5000/v1`
3. API Key: любой (proxy не проверяет)
4. Connect

**Преимущества proxy:**
- Автоматическое сохранение сообщений в память
- Извлечение релевантных воспоминаний
- Инжекция контекста в промпт
- Изоляция памяти по персонажам

### Вариант 3: Text Completion API Type

1. SillyTavern → API Connections → **API Type**: `Text Completion` или `KoboldAI`
2. Base URL: `http://127.0.0.1:8080` (monolith) или `http://127.0.0.1:5000` (proxy)
3. Connect

## ⚠️ FIX: Port Mismatch Error

The error you're seeing:
```
:8000/api/backends/text-completions/status:1 Failed to load resource
```

This means SillyTavern is configured for port **8000**, but RTMDK server runs on port **8080**.

### How to Fix

1. In SillyTavern → API Connections
2. Change the port from `8000` to `8080` (monolith) or `5000` (proxy)
3. Make sure Base URL is: `http://127.0.0.1:8080` or `http://127.0.0.1:5000`

### Manual Tests
```bash
# Test health
curl http://127.0.0.1:8080/health

# Test status (SillyTavern format)
curl -X POST http://127.0.0.1:8080/api/backends/text-completions/status ^
  -H "Content-Type: application/json" ^
  -d "{}"

# Test text generation (SillyTavern format)
curl -X POST http://127.0.0.1:8080/api/v1/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Hello\",\"max_new_tokens\":50,\"stream\":false}"
```

## Common Issues & Solutions

### Issue 1: "Failed to connect" or 400 Bad Request
**Cause**: Wrong port (8000 vs 8080)
**Solution**: Change SillyTavern to use port 8080

### Issue 2: Server not running
**Solution**: Run `python rtmdk_server.py`

### Issue 3: Streaming doesn't work
**Solution**: Try disabling streaming in SillyTavern settings

## Web UI Dashboard

Access at: `http://127.0.0.1:8080/dashboard`

Features:
- Model selection (LM Studio, OpenRouter, OpenAI, Anthropic)
- Embedder selection
- Memory statistics
- Backup/Restore
- UX feature toggles
