# RTMDK Docker + Silly Tavern + External APIs Setup Guide

> Полная инструкция по запуску RTMDK через Docker с внешними API и Silly Tavern

---

## Часть 1: Запуск RTMDK через Docker

### 1.1. Быстрый старт (3 команды)

```bash
# 1. Скопируйте .env.example в .env
cp .env.example .env

# 2. Настройте .env (выберите API провайдера — см. ниже)

# 3. Запустите контейнер
docker-compose up -d

# 4. Проверьте работоспособность
curl http://localhost:8080/health
```

**Ожидаемый ответ:**
```json
{
  "status": "ok",
  "version": "8.0.0",
  "api_provider": "openrouter",
  "memory_nodes": 0
}
```

### 1.2. Настройка .env — Выбор API провайдера

Откройте `.env` и настройте **один** из провайдеров:

#### Вариант A: LM Studio (локально, бесплатно)
```bash
RTMDK_API_PROVIDER=lm_studio
LM_STUDIO_URL=http://host.docker.internal:12345/v1
RTMDK_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5-GGUF
```

#### Вариант B: OpenRouter (унифицированный API, много моделей)
```bash
RTMDK_API_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-ваш_ключ
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

#### Вариант C: OpenAI (официальный API)
```bash
RTMDK_API_PROVIDER=openai
OPENAI_API_KEY=sk-proj-ваш_ключ
OPENAI_MODEL=gpt-4o
```

#### Вариант D: Anthropic (официальный API)
```bash
RTMDK_API_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-ваш_ключ
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

#### Вариант E: Любой OpenAI-совместимый API (Groq, Together AI, LocalAI)
```bash
RTMDK_API_PROVIDER=custom
CUSTOM_API_URL=https://api.groq.com/openai/v1
CUSTOM_API_KEY=gsk-ваш_ключ
CUSTOM_API_MODEL=llama-3.1-70b-versatile
```

### 1.3. Все переменные окружения

| Переменная | По умолчанию | Описание |
|-----------|:---:|---|
| `RTMDK_HOST` | `0.0.0.0` | Хост сервера |
| `RTMDK_PORT` | `8080` | Порт API |
| `RTMDK_API_KEY` | `rtmdk-local` | API ключ для аутентификации |
| `RTMDK_LOG_FORMAT` | `text` | Формат логов: `text` или `json` |
| `RTMDK_MEMORY_FILE` | `/data/memory.json` | Путь к файлу памяти внутри контейнера |
| `RTMDK_AUTO_SAVE` | `60` | Интервал автосохранения (сек) |
| `RTMDK_API_PROVIDER` | `lm_studio` | Провайдер: `lm_studio`, `openrouter`, `openai`, `anthropic`, `custom` |
| `RTMDK_API_TIMEOUT` | `30` | Таймаут запросов к API (сек) |
| `LM_STUDIO_URL` | `http://host.docker.internal:12345/v1` | URL LM Studio API |
| `RTMDK_EMBED_MODEL` | `nomic-embed-text-v1.5-GGUF` | Модель эмбеддингов |
| `OPENROUTER_API_KEY` | — | Ключ OpenRouter |
| `OPENROUTER_MODEL` | `anthropic/claude-3.5-sonnet` | Модель OpenRouter |
| `OPENAI_API_KEY` | — | Ключ OpenAI |
| `OPENAI_MODEL` | `gpt-4o` | Модель OpenAI |
| `ANTHROPIC_API_KEY` | — | Ключ Anthropic |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Модель Anthropic |
| `CUSTOM_API_URL` | — | URL кастомного API |
| `CUSTOM_API_KEY` | — | Ключ кастомного API |
| `CUSTOM_API_MODEL` | — | Модель кастомного API |

### 1.4. Полезные команды

```bash
# Запуск (CPU)
docker-compose up -d

# Запуск (GPU — раскомментируйте секцию rtmdk-api-gpu в docker-compose.yml)
# docker-compose up -d rtmdk-api-gpu

# Остановка
docker-compose down

# Логи
docker-compose logs -f

# Перезапуск
docker-compose restart

# Удаление с потерей данных
docker-compose down -v

# Просмотр состояния
docker-compose ps

# Проверка здоровья
curl http://localhost:8080/health
```

### 1.5. Доступные API Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус сервера |
| GET | `/v1/models` | Список моделей |
| POST | `/v1/chat/completions` | Чат с памятью (OpenAI-compatible) |
| POST | `/v1/embeddings` | Эмбеддинги |
| GET | `/v1/memory/stats` | Статистика памяти |
| GET | `/v1/memory/health` | Здоровье поля |
| POST | `/v1/memory/save` | Сохранить контекст |
| POST | `/v1/memory/query` | Запросить память |
| POST | `/v1/memory/clear` | Очистить память |

### 1.6. Запуск с GPU

Для GPU-версии:

1. Откройте `docker-compose.yml`
2. Раскомментируйте секцию `rtmdk-api-gpu`
3. Закомментируйте секцию `rtmdk-api` (CPU)
4. Убедитесь что установлен NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

5. Запустите:
```bash
docker-compose up -d rtmdk-api-gpu
```

**Проверка GPU:**
```bash
docker exec rtmdk-memory-gpu nvidia-smi
```

**Требования для GPU:**
- NVIDIA GPU с поддержкой CUDA 12.1+
- NVIDIA Container Toolkit установлен
- Docker с поддержкой GPU

### 1.7. Сравнение CPU vs GPU

| Параметр | CPU Docker | GPU Docker |
|----------|:---:|:---:|
| Размер образа | ~200 MB | ~4 GB |
| RAM контейнера | ~200 MB | ~2 GB |
| VRAM | 0 MB | ~50-200 MB |
| Скорость (батч 1000) | ~150ms | ~15ms |
| Скорость (1 запрос) | ~25ms | ~8ms |

**Когда использовать GPU:**
- High-throughput сервер (>100 запросов/сек)
- Батчевая обработка эмбеддингов
- Нейронное ОДУ (continuous dynamics)

**Когда достаточно CPU:**
- Персональный ассистент (<10 запросов/сек)
- Тестирование и разработка
- Обычное использование RTMDK

---

## Часть 2: Интеграция с Silly Tavern

### 2.1. Что такое Silly Tavern

Silly Tavern — это фронтенд для общения с LLM-персонажами. Поддерживает:
- Загрузку карточек персонажей
- Управление чатом и памятью
- Интеграцию с OpenAI-compatible API
- Vector Storage extension для RAG

### 2.2. Способ 1: Через OpenAI API (рекомендуется)

Этот способ позволяет RTMDK обрабатывать запросы и возвращать ответы через Silly Tavern.

**Шаг 1: Настройте Silly Tavern**

1. Откройте Silly Tavern → Settings → API Connection
2. Выберите API Type: **OpenAI**
3. Установите:
   - **OpenAI Base URL:** `http://localhost:8080/v1`
   - **API Key:** `rtmdk-local`
   - **Model:** `rtmdk`

4. Нажмите **Connect** — должно показать "Connected"

**Шаг 2: Настройте персонажа**

Создайте или загрузите карточку персонажа как обычно. RTMDK автоматически будет:
1. Сохранять контекст диалога в память
2. Извлекать релевантные воспоминания при новых сообщениях
3. Включать воспоминания в системный промпт для LLM

**Шаг 3: Проверьте работу**

Напишите персонажу что-то вроде:
> "Меня зовут Никита, я люблю кофе по утрам"

Затем через несколько сообщений спросите:
> "Как меня зовут и что я люблю по утрам?"

Персонаж должен вспомнить из контекста RTMDK.

### 2.3. Способ 2: Через Vector Storage extension (RAG)

Этот способ использует RTMDK как источник контекста, а LLM отвечает через отдельный API.

**Шаг 1: Настройте основной API**

В Silly Tavern → Settings → API Connection:
- Настройте ваш основной LLM (например, LM Studio на порту 12345)

**Шаг 2: Установите Vector Storage extension**

1. Extensions → Download Extensions and Assets
2. Найдите **Vector Storage** и установите
3. Перезапустите Silly Tavern

**Шаг 3: Настройте Vector Storage**

1. Extensions → Vector Storage
2. Установите:
   - **API Endpoint:** `http://localhost:8080/v1`
   - **API Key:** `rtmdk-local`
   - **Collection:** `rtmdk-memory`
   - **Top K:** `5`
3. Включите **"Inject memories into context"**

**Шаг 4: Проверьте работу**

Теперь при каждом сообщении Silly Tavern будет:
1. Отправлять запрос в RTMDK для поиска релевантных воспоминаний
2. Вставлять воспоминания в контекст
3. Отправлять контекст + запрос в вашу LLM

### 2.4. Способ 3: Через World Info (Advanced)

Для продвинутых пользователей — интеграция через World Info.

**Шаг 1: Создайте скрипт-посредник**

Создайте файл `rtmdk_bridge.py`:

```python
#!/usr/bin/env python3
"""Bridge between Silly Tavern World Info and RTMDK."""

import requests
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

RTMDK_URL = "http://localhost:8080/v1"
API_KEY = "rtmdk-local"

@app.route("/query", methods=["POST"])
def query_memory():
    """Query RTMDK memory for relevant context."""
    data = request.json
    user_input = data.get("text", "")

    resp = requests.post(
        f"{RTMDK_URL}/memory/query",
        json={"query": user_input, "session_id": "silly_tavern"},
        headers={"Authorization": f"Bearer {API_KEY}"}
    )

    if resp.status_code == 200:
        context = resp.json().get("context", "")
        return jsonify({"entries": [{"keys": [], "content": context, "constant": False, "selective": True, "insertion_order": 100}]})
    return jsonify({"entries": []})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5100)
```

**Шаг 2: Настройте World Info**

1. Silly Tavern → World Info → Add New
2. URL: `http://localhost:5100/query`
3. Method: POST
4. Включите "Activate on user input"

---

## Часть 3: Типовые сценарии использования

### 3.1. Персонаж с долгосрочной памятью

**Сценарий:** Вы общаетесь с персонажем днями/неделями, он помнит ключевые детали.

**Настройка:**
```bash
# .env
RTMDK_AUTO_SAVE=60          # Сохранять каждую минуту
RTMDK_ENABLE_LM_STUDIO=true # Реальные эмбеддинги
```

**Результат:** Персонаж помнит ваше имя, предпочтения, историю разговоров.

### 3.2. Мультиперсонажная память

**Сценарий:** Несколько персонажей, каждый со своей памятью.

**Настройка:**
```bash
# Используйте разные session_id для разных персонажей
# В Silly Tavern это настраивается через Character Notes или скрипты
```

### 3.3. Корпоративный ассистент с базой знаний

**Сценарий:** Ассистент отвечает на вопросы по документации компании.

**Настройка:**
1. Загрузите документы в RTMDK через API:
```bash
curl -X POST http://localhost:8080/v1/memory/save \
  -H "Content-Type: application/json" \
  -d '{"input": "Как оформить отпуск?", "output": "Для оформления отпуска обратитесь в HR..."}'
```

2. Настройте Silly Tavern на использование RTMDK API
3. Ассистент будет отвечать на основе загруженных документов

---

## Часть 4: Troubleshooting

| Проблема | Причина | Решение |
|----------|---------|---------|
| `curl: (7) Failed to connect` | Контейнер не запущен | `docker-compose up -d` |
| `lm_studio: false` в health | LM Studio не доступен | Проверьте `LM_STUDIO_URL` и запустите LM Studio |
| Пустые ответы | Нет памяти в RTMDK | Отправьте несколько запросов через `/v1/memory/save` |
| Silly Tavern не подключается | Неверный URL/API key | Проверьте `http://localhost:8080/v1` и ключ `rtmdk-local` |
| Медленные ответы | LM Studio на CPU | Загрузите меньшую модель или используйте GPU |

---

## Часть 5: Production Checklist

- [ ] Установите `RTMDK_API_KEY` в надёжное значение
- [ ] Включите `RTMDK_LOG_FORMAT=json` для продакшена
- [ ] Настройте `RTMDK_AUTO_SAVE=30` для частого сохранения
- [ ] Настройте volume backup для `/data/memory.json`
- [ ] Мониторьте `/health` endpoint
- [ ] Настройте лимиты памяти в docker-compose

---

*Последнее обновление: Апрель 2026, RTMDK v8.0*
