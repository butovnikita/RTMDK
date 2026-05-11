# RTMDK — Локальный запуск

> Запуск RTMDK на своём ПК — 3 варианта: Docker, Python, SillyTavern

---

## Быстрый старт за 5 минут

### Вариант A: Python (рекомендуется для разработки)

```bash
cd C:\Users\Никита\Desktop\llm_lab
pip install -r requirements-home.txt
python rtmdk_server.py
```

Сервер запущен на `http://localhost:8080`

### Вариант B: Docker Production

```bash
docker-compose -f docker-compose.prod.yml up -d
curl http://localhost:8080/health
```

### Вариант C: Docker Home + SillyTavern

```bash
docker-compose -f docker-compose.home.yml up -d
# Сервер: http://localhost:8080
# SillyTavern Proxy: http://localhost:5000
```

---

## Предварительные требования

| Компонент | Зачем | Обязательно? |
|-----------|-------|-------------|
| Python 3.10+ | Запуск сервера | ✅ Да |
| LM Studio | Реальные LLM + эмбеддинги | ❌ Опционально |
| Docker Desktop | Контейнеризация | ❌ Опционально |

---

## Конфигурация через пресеты

```bash
# Локальный ассистент (по умолчанию)
python rtmdk_server.py

# Production сервер
RTMDK_PRESET=production python rtmdk_server.py

# Research режим
RTMDK_PRESET=research python rtmdk_server.py

# С кастомными параметрами
RTMDK_PRESET=local RTMDK_LATENT_DIM=128 RTMDK_TOP_K=10 python rtmdk_server.py
```

---

## Интеграция с LM Studio

1. **Запусти LM Studio:**
   - Загрузи модель (например, Qwen2.5-7B)
   - Включи сервер: Server → Start на порту `12345`
   - Загрузи модель эмбеддингов: `nomic-embed-text-v1.5`

2. **Запусти RTMDK:**
   ```bash
   python rtmdk_server.py
   ```

3. **Проверь интеграцию:**
   ```bash
   curl http://localhost:8080/health
   # → "lm_studio": true — значит подключено!
   ```

---

## Использование API

### OpenAI-compatible клиент

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="rtmdk-local"
)

response = client.chat.completions.create(
    model="rtmdk",
    messages=[{"role": "user", "content": "Привет, это мой тестовый запрос"}],
)
print(response.choices[0].message.content)
```

### Сохранить память

```bash
curl -X POST http://localhost:8080/v1/memory/nodes \
  -H "Content-Type: application/json" \
  -d '{"content":"Меня зовут Никита","salience":0.8}'
```

### Запросить память

```bash
curl -X POST http://localhost:8080/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query":"как меня зовут?","session_id":"default"}'
```

### Статистика и здоровье

```bash
curl http://localhost:8080/health
curl http://localhost:8080/health/deep
curl http://localhost:8080/metrics      # Prometheus metrics
curl http://localhost:8080/dashboard    # Веб-UI (если доступен)
```

---

## React Admin Panel

Визуальный интерфейс управления памятью.

### Запуск

```bash
cd admin
npm install
npm run dev
```

Открой `http://localhost:5173` (URL покажет терминал).

### Вкладки

- **Dashboard** — health status, количество нод, версия
- **Memory Nodes** — пагинированная таблица всех узлов памяти
- **Query** — интерактивный поиск по памяти с live-результатами
- **SOT** — статус Self-Organizing Tokenizer и просмотр словаря

### Production build

```bash
cd admin
npm run build
# Статические файлы в dist/
```

---

## SillyTavern интеграция

### Вариант 1: Monolith (проще)

```bash
python rtmdk_server.py
# SillyTavern → API Type: OpenAI → Base URL: http://localhost:8080/v1
```

### Вариант 2: Proxy (рекомендуется)

```bash
python rtmdk_sillytavern_launcher.py
# Запускает сервер (8080) + proxy (5000)
# SillyTavern → API Type: OpenAI → Base URL: http://localhost:5000/v1
```

---

## Docker

### Production (без SillyTavern)

```bash
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml logs -f
docker-compose -f docker-compose.prod.yml down
```

### Home + SillyTavern

```bash
docker-compose -f docker-compose.home.yml up -d
# Два контейнера: rtmdk-home-server (8080) + rtmdk-home-proxy (5000)
```

### С LM Studio из хоста

```yaml
# docker-compose.override.yml
services:
  rtmdk:
    environment:
      - LM_STUDIO_URL=http://host.docker.internal:12345/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

---

## Управление контейнером

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
docker-compose -f docker-compose.prod.yml logs -f rtmdk

# Войти в контейнер
docker exec -it rtmdk-server sh
```

---

## Переменные окружения

```yaml
environment:
  - RTMDK_HOST=0.0.0.0
  - RTMDK_PORT=8080
  - RTMDK_MEMORY_FILE=/data/memory.json
  - RTMDK_PRESET=local          # Пресет: local/production/research/...
  - RTMDK_LATENT_DIM=64         # Переопределение параметра
  - RTMDK_DECAY_RATE=0.997
  - RTMDK_ENABLE_LM_STUDIO=true
  - LM_STUDIO_URL=http://host.docker.internal:12345/v1
  - RTMDK_AUTO_SAVE=60
  - RTMDK_API_KEY=rtmdk-local
```

---

## Файлы памяти

| Файл | Описание |
|------|----------|
| `~/.rtmdk/memory.json` | Автосохранение (каждые 60 сек) |
| `~/.rtmdk/backups/` | Бэкапы памяти |
| `.env` | Персистентная конфигурация |

---

## Отладка

```bash
# Проверить здоровье
curl http://localhost:8080/health

# Smoke test
python tests/smoke_test.py

# Python REPL
from rtmdk import RTMDKMemory, RTMDKConfig
config = RTMDKConfig.local()
memory = RTMDKMemory(config=config, embedder=my_embedder)
```

### Pre-commit hooks (форматирование)

Установи хуки для автоматической проверки формата перед коммитом:

```bash
pip install pre-commit
pre-commit install
```

Настроены: `black`, `isort`, базовые проверки (`trailing-whitespace`, `check-yaml`).
Запустить вручную на всех файлах:

```bash
pre-commit run --all-files
```

### Частые проблемы

| Проблема | Решение |
|----------|---------|
| Port 8080 занят | `RTMDK_PORT=8081 python rtmdk_server.py` |
| LM Studio не подключается | Проверь что сервер запущен на 12345 |
| Память не сохраняется | Проверь права на `~/.rtmdk/` |
| Docker не запускается | `docker-compose down -v && docker-compose up -d --build` |

---

*Инструкция актуальна для RTMDK v8.3 с unified config architecture.*
