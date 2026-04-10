# Quick Start Guide — RTMDK + SillyTavern

## The Easy Way (Recommended)

### 1. Start Everything
```bash
python rtmdk_sillytavern_launcher.py
```
Or on Windows, double-click `start_sillytavern.bat`

This starts:
- RTMDK Server (memory, port 8080)
- SillyTavern Proxy (port 5000)

### 2. Configure SillyTavern
1. Open SillyTavern → Extensions → API Connections
2. Select **API Type**: `OpenAI`
3. Configure:
   - **Base URL**: `http://127.0.0.1:5000/v1`
   - **API Key**: `anything`
   - **Model**: `anything`
4. Click **Connect**

### 3. Start Chatting
- Your messages are automatically saved to memory
- The AI will remember past conversations
- Each character has separate memory

---

## The Manual Way

### 1. Start RTMDK Server
```bash
python rtmdk_server.py
```

### 2. Start Proxy (in new terminal)
```bash
python rtmdk_st_proxy.py
```

### 3. Configure SillyTavern
Same as above.

---

## Configuration

Edit `st_config.json`:
```json
{
  "memory": {
    "save_user_messages": true,
    "save_ai_messages": true,
    "session_per_character": true
  },
  "retrieval": {
    "max_memories": 3,
    "memory_format": "narrative"
  }
}
```

---

## Troubleshooting

**Problem:** SillyTavern can't connect
**Solution:** Make sure launcher is running and Proxy is on port 5000

**Problem:** No memories being saved
**Solution:** Check that RTMDK Server is running on port 8080

**Problem:** LM Studio errors
**Solution:** Start LM Studio and load a model before using SillyTavern

---

## Commands

```bash
# Show status
python rtmdk_sillytavern_launcher.py --status

# Custom ports
python rtmdk_sillytavern_launcher.py --rtmdk-port 9090 --proxy-port 6000

# Stop services
Ctrl+C in the launcher window
```
