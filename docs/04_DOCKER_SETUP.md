# RTMDK Docker + SillyTavern Setup Guide v8.1

> Полная инструкция по запуску RTMDK через Docker с unified config и SillyTavern

---

## Часть 1: Быстрый старт (3 команды)

### Production (без SillyTavern)

```bash
docker-compose -f docker-compose.prod.yml up -d
curl http://localhost:8080/health
```

### Home + SillyTavern

```bash
docker-compose -f docker-compose.home.yml up -d
curl http://localhost:8080/health        # Сервер
curl http://localhost:5000/health        # SillyTavern Proxy
```

**Ожидаемый ответ:**
```json
{"status": "ok", "version": "8.0.0", "lm_studio": true, "memory_nodes": 0}
```

---

## Часть 2: Выбор конфигурации

| Файл | Назначение | Порты | SillyTavern |
|------|-----------|-------|-------------|
| `docker-compose.prod.yml` | Production API | 8080 | ❌ |
| `docker-compose.home.yml` | Home + SillyTavern | 8080 + 5000 | ✅ |

### Dockerfile'ы

| Файл | Описание | Размер |
|------|---------|--------|
| `Dockerfile` | Production (min dependencies) | ~200MB |
| `Dockerfile.home` | Home (all features + ST proxy) | ~400MB |
| `Dockerfile.gpu` | GPU (CUDA 12.1) | ~4GB |

---

## Часть 3: Настройка .env

### Выбор API провайдера

#### Вариант A: LM Studio (локально, бесплатно)
```bash
RTMDK_API_PROVIDER=lm_studio
LM_STUDIO_URL=http://host.docker.internal:12345/v1
```

#### Вариант B: OpenRouter (много моделей)
```bash
RTMDK_API_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-ваш_ключ
```

#### Вариант C: OpenAI
```bash
RTMDK_API_PROVIDER=openai
OPENAI_API_KEY=sk-proj-ваш_ключ
```

### Выбор пресета

```bash
RTMDK_PRESET=local          # По умолчанию
RTMDK_PRESET=production     # Для сервера
RTMDK_PRESET=research       # Для экспериментов
```

### Переопределение параметров

```bash
RTMDK_LATENT_DIM=128
RTMDK_TOP_K=10
RTMDK_DECAY_RATE=0.9995
```

---

## Часть 4: Интеграция с LM Studio

Для доступа к LM Studio на хосте из Docker:

```yaml
# docker-compose.override.yml
services:
  rtmdk:
    environment:
      - LM_STUDIO_URL=http://host.docker.internal:12345/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Перезапусти:
```bash
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d
```

Проверь:
```bash
curl http://localhost:8080/health
# → "lm_studio": true
```

---

## Часть 5: SillyTavern подключение

### Через Proxy (рекомендуется)

```
SillyTavern → OpenAI API → http://localhost:5000/v1 → API Key: любой
```

Proxy автоматически:
- Сохраняет сообщения пользователя и AI в память
- Извлекает релевантные воспоминания
- Инжектирует контекст в промпт
- Изолирует память по персонажам

### Через Monolith

```
SillyTavern → OpenAI API → http://localhost:8080/v1 → API Key: rtmdk-local
```

---

## Часть 6: Управление

```bash
# Запуск
docker-compose -f docker-compose.prod.yml up -d

# Остановка
docker-compose -f docker-compose.prod.yml down

# Остановка с удалением данных
docker-compose -f docker-compose.prod.yml down -v

# Пересборка
docker-compose -f docker-compose.prod.yml up -d --build

# Логи
docker-compose -f docker-compose.prod.yml logs -f

# Войти в контейнер
docker exec -it rtmdk-server sh

# Проверить здоровье
docker-compose -f docker-compose.prod.yml ps
curl http://localhost:8080/health
```

---

## Часть 7: API Examples

### Сохранить память

```bash
curl -X POST http://localhost:8080/v1/memory/save \
  -H "Content-Type: application/json" \
  -d '{"input":"Меня зовут Никита","output":"Запомнил"}'
```

### Запросить память

```bash
curl -X POST http://localhost:8080/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query":"как меня зовут?"}'
```

### Изменить конфигурацию

```bash
# Сменить пресет (требует перезапуска)
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"RTMDK_PRESET": "production"}'

# Сменить модель эмбеддера (применяется сразу)
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"RTMDK_EMBED_MODEL": "text-embedding-3-small"}'
```

---

## Часть 8: Отладка

### Логи

```bash
docker-compose -f docker-compose.prod.yml logs -f
```

### Smoke test

```bash
python tests/smoke_test.py
```

### Python REPL

```python
from rtmdk import RTMDKMemory, RTMDKConfig

# С пресетом
config = RTMDKConfig.local()
memory = RTMDKMemory(config=config, embedder=my_embedder)

# С env var overrides
import os
os.environ['RTMDK_LATENT_DIM'] = '128'
config = RTMDKConfig.local()  # latent_dim=128
```

### Частые проблемы

| Проблема | Решение |
|----------|---------|
| Port 8080 занят | `"8081:8080"` в docker-compose |
| LM Studio не подключается | Проверь `host.docker.internal` в override |
| Память не сохраняется | Проверь volume `rtmdk-data:/data` |
| Docker не запускается | `docker-compose down -v && docker-compose up -d --build` |

---

*Инструкция актуальна для RTMDK v8.1 с unified config architecture.*
