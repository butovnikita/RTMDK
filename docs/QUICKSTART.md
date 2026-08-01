# Quick Start Guide — RTMDK + SillyTavern

## The Easy Way (Recommended)

### 1. Start Everything
```bash
python legacy/rtmdk_sillytavern_launcher.py
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
python legacy/rtmdk_server.py
```

### 2. Start Proxy (in new terminal)
```bash
python legacy/rtmdk_st_proxy.py
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
python legacy/rtmdk_sillytavern_launcher.py --status

# Custom ports
python legacy/rtmdk_sillytavern_launcher.py --rtmdk-port 9090 --proxy-port 6000

# Stop services
Ctrl+C in the launcher window
```

## Конфигурация через пресеты

```bash
# Выбрать пресет
RTMDK_PRESET=local python legacy/rtmdk_sillytavern_launcher.py

# С кастомными параметрами
RTMDK_PRESET=local RTMDK_LATENT_DIM=128 python legacy/rtmdk_sillytavern_launcher.py
```

Доступные пресеты: `local`, `production`, `research`, `enterprise`, `agent`, `legal`, `medical`, `streaming`.

## Mathematical Enhancement Track (P0–P2)

Включается через переменные окружения:

```bash
# P0.1 — Риманова геометрия (Пуанкаре)
RTMDK_HYPERBOLIC=true RTMDK_BALL_RADIUS=0.85

# P1.2 — Локальная адаптивная ширина ядра
RTMDK_ADAPTIVE_BANDWIDTH=true RTMDK_ADAPTIVE_BANDWIDTH_K=5

# P1.1 — Конформальное предсказание
RTMDK_CONFORMAL_PREDICTION=true RTMDK_CONFORMAL_ALPHA=0.10

# P2.1 — Спектральная кластеризация
RTMDK_SPECTRAL_CONSOLIDATION=true RTMDK_SPECTRAL_MAX_CLUSTERS=10

# P2.2 — Фильтр Калмана
RTMDK_ENABLE_KALMAN_FILTER=true RTMDK_KALMAN_DIAGONAL_APPROX=true
```

Все фичи отключены по умолчанию (`False`). Включайте по одной для оценки влияния на ваш сценарий.
