# SillyTavern Setup Guide with RTMDK Proxy

## Architecture

```
SillyTavern ←→ RTMDK Proxy (port 5000) ←→ LM Studio (port 12345)
                        ↓
                  RTMDK Server (port 8080)
                        ↓
                  memory.json
```

## Setup Instructions

### 1. Start RTMDK Server
```bash
python rtmdk_server.py
```

### 2. Start RTMDK Proxy
```bash
python rtmdk_st_proxy.py
```

### 3. Configure SillyTavern

1. Open SillyTavern → Extensions → API Connections
2. Select **API Type**: `OpenAI`
3. Configure:
   - **Base URL**: `http://127.0.0.1:5000/v1`
   - **API Key**: `anything` (not checked by proxy)
   - **Model**: Any model name (proxy forwards to LM Studio)
4. Click **Connect**

### 4. How It Works

1. **You send a message** in SillyTavern
2. **Proxy saves** your message to RTMDK memory
3. **Proxy queries** RTMDK for relevant past memories
4. **Proxy injects** memories into the system prompt
5. **LM Studio generates** response with memory context
6. **Proxy saves** the AI response back to memory
7. **Response returns** to SillyTavern

### 5. Configuration

Edit `st_config.json`:

```json
{
  "memory": {
    "save_user_messages": true,    // Save your messages to memory
    "save_ai_messages": true,      // Save AI responses to memory
    "session_per_character": true  // Separate memory per character
  },
  "retrieval": {
    "max_memories": 3,             // How many memories to retrieve
    "memory_format": "narrative"   // "narrative" or "bullet_points"
  }
}
```

### 6. Verify It's Working

Check proxy status:
```bash
curl http://127.0.0.1:5000/status
```

Check memory stats:
```bash
curl http://127.0.0.1:5000/memory/stats
```

### 7. Troubleshooting

**Problem:** SillyTavern can't connect
**Solution:** Make sure proxy is running on port 5000

**Problem:** No memories being saved
**Solution:** Check that RTMDK server is running on port 8080

**Problem:** Memories not affecting responses
**Solution:** Check `st_config.json` has `retrieval.enabled: true`
