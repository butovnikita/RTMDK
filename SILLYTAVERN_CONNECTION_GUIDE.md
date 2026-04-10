# RTMDK SillyTavern Connection Guide

## Server Status: WORKING ✅

All endpoints tested and verified:
- `/health` → Returns server health
- `/v1/models` → Returns available models  
- `/v1/chat/completions` → OpenAI chat format (WORKS)
- `/v1/embeddings` → Embedding endpoint (WORKS)
- `/api/v1/generate` → SillyTavern text completion format (WORKS)
- `/api/backends/text-completions/generate` → ST backend format (WORKS)
- `/api/backends/text-completions/status` → ST status check (WORKS)
- `/dashboard` → Web UI (WORKS)

## ⚠️ FIX: Port Mismatch Error

The error you're seeing:
```
:8000/api/backends/text-completions/status:1 Failed to load resource
```

This means SillyTavern is configured for port **8000**, but RTMDK server runs on port **8080**.

### How to Fix

1. In SillyTavern → API Connections
2. Change the port from `8000` to `8080`
3. Make sure Base URL is: `http://127.0.0.1:8080`

## SillyTavern Configuration Options

### Option 1: Text Completion API Type (RECOMMENDED for RTMDK)

1. Open SillyTavern → Extensions → API Connections
2. Select **API Type**: `Text Completion` or `KoboldAI`
3. Configure:
   - **Base URL**: `http://127.0.0.1:8080`
   - **API Key**: `rtmdk-local` (or leave empty)
4. Click **Connect** - should show "Connected"

### Option 2: OpenAI API Type

1. Select **API Type**: `OpenAI`
2. Configure:
   - **Base URL**: `http://127.0.0.1:8080/v1`
   - **API Key**: `rtmdk-local` (or leave empty)
   - **Model**: `rtmdk` or select from list
3. Click **Connect** - should show "Connected"

## Testing Connection

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
