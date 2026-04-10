# RTMDK SillyTavern Connection Guide

## Server Status: WORKING ✅

All endpoints tested and verified:
- `/health` → Returns server health
- `/v1/models` → Returns available models  
- `/v1/chat/completions` → OpenAI chat format (WORKS)
- `/v1/embeddings` → Embedding endpoint (WORKS)
- `/api/v1/generate` → SillyTavern text completion format (WORKS)
- `/v1/completions` → OpenAI completions format (WORKS)
- `/dashboard` → Web UI (WORKS)

## SillyTavern Configuration

### Option 1: OpenAI API Type (RECOMMENDED)

1. Open SillyTavern → Extensions → API Connections
2. Select **API Type**: `OpenAI`
3. Configure:
   - **Base URL**: `http://127.0.0.1:8080/v1`
   - **API Key**: `rtmdk-local` (or leave empty)
   - **Model**: `rtmdk` or select from list
4. Click **Connect** - should show "Connected"

### Option 2: Text Completion API Type

1. Select **API Type**: `Text Completion`
2. Configure:
   - **Base URL**: `http://127.0.0.1:8080`
   - **API Key**: `rtmdk-local`
3. Click **Connect**

## Common Issues & Solutions

### Issue 1: "Failed to connect"
**Cause**: Server not running
**Solution**: Run `python rtmdk_server.py`

### Issue 2: "Model not found"
**Cause**: SillyTavern looking for specific model
**Solution**: The server accepts any model name, but shows LM Studio models

### Issue 3: "401 Unauthorized"
**Cause**: Wrong API key
**Solution**: Use `rtmdk-local` or check your `.env` file

### Issue 4: Streaming doesn't work
**Cause**: SillyTavern expects specific streaming format
**Solution**: Try disabling streaming in SillyTavern settings

## Testing Connection

### Manual Test
```bash
# Test health
curl http://127.0.0.1:8080/health

# Test models
curl http://127.0.0.1:8080/v1/models

# Test chat (OpenAI format)
curl -X POST http://127.0.0.1:8080/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"rtmdk\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"stream\":false}"

# Test text generation (SillyTavern format)
curl -X POST http://127.0.0.1:8080/api/v1/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"prompt\":\"Hello\",\"max_new_tokens\":50,\"stream\":false}"
```

## Web UI Dashboard

Access at: `http://127.0.0.1:8080/dashboard`

Features:
- Model selection (LM Studio, OpenRouter, OpenAI, Anthropic)
- Embedder selection
- Memory statistics
- Backup/Restore
- UX feature toggles

## Server Logs

Check `server_test2.log` for startup logs and errors.
