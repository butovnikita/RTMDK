# RTMDK Docker + Silly Tavern Setup Guide v8.0

> Полная инструкция по запуску RTMDK через Docker с 27 UX-функциями и Silly Tavern

---

## Часть 1: Быстрый старт (3 команды)

```bash
cp .env.example .env
docker-compose up -d
curl http://localhost:8080/health
```

**Ожидаемый ответ:**
```json
{"status": "ok", "version": "8.0.0", "api_provider": "lm_studio", "memory_nodes": 0}
```

---

## Часть 2: Настройка .env

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
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
```

#### Вариант C: OpenAI
```bash
RTMDK_API_PROVIDER=openai
OPENAI_API_KEY=sk-proj-ваш_ключ
OPENAI_MODEL=gpt-4o
```

---

## Часть 3: UX Features в Docker

### Все 27 UX-функций доступны через API

| Категория | Эндпоинты | Что делает |
|-----------|-----------|---|
| **Feedback** | `POST /v1/feedback` | Thumbs up/down для улучшения памяти |
| **Sessions** | `POST /v1/session/save` | Сохранение/загрузка сессий пользователей |
| **Backups** | `POST /v1/backup/create` | Автобэкапы с ротацией |
| **Import** | `POST /v1/import/json` | Импорт JSON/CSV/URL |
| **Analytics** | `GET /v1/analytics` | Статистика и тренды памяти |
| **Health** | `GET /v1/health`, `GET /v1/metrics` | Мониторинг + Prometheus |
| **Export** | `GET /v1/export?format=md` | Экспорт в Markdown/Text/JSON |
| **Tags** | `POST /v1/tags/{node_id}` | Кастомные теги на узлах |
| **Rate Limit** | `GET /v1/rate-limit` | Статус лимитов |
| **Events** | `GET /v1/events` | SSE поток событий |
| **Cache** | `GET /v1/cache/stats` | Статистика кэша эмбеддингов |

### Docker volumes для persistency

```yaml
volumes:
  rtmdk-data:/data              # Основная память
  rtmdk-backups:/data/backups   # Бэкапы (переживают rebuild)
  rtmdk-sessions:/data/sessions # Сессии пользователей
  rtmdk-cache:/data/embedding_cache  # Кэш эмбеддингов
```

### UX Environment Variables

| Переменная | По умолчанию | Описание |
|-----------|:---:|---|
| `RTMDK_BACKUP_ROTATION` | 5 | Кол-во бэкапов для ротации |
| `RTMDK_AUTO_SAVE_INTERVAL` | 60 | Автосохранение (сек) |
| `RTMDK_PRUNE_AGE_DAYS` | 90 | Возраст для pruning (дни) |
| `RTMDK_PRUNE_MIN_SALIENCE` | 0.05 | Мин. salience для pruning |
| `RTMDK_RATE_LIMIT_PER_MINUTE` | 60 | Лимит запросов/мин |
| `RTMDK_CACHE_MAX_SIZE` | 100000 | Макс. размер кэша |
| `RTMDK_MAX_CONTEXT_TOKENS` | 300 | Макс. токенов контекста |
| `RTMDK_FEEDBACK_LR` | 0.05 | Learning rate feedback |

---

## Часть 4: Интеграция с Silly Tavern

### Способ 1: OpenAI API (рекомендуется)

1. Silly Tavern → Settings → API Connection
2. API Type: **OpenAI**
3. OpenAI Base URL: `http://localhost:8080/v1`
4. API Key: `rtmdk-local`
5. Model: `rtmdk`

### Способ 2: Vector Storage extension

1. Extensions → Download → **Vector Storage**
2. API Endpoint: `http://localhost:8080/v1`
3. API Key: `rtmdk-local`
4. ✓ Inject memories into context

---

## Часть 5: Запуск с GPU

Для GPU-версии раскомментируйте секцию `rtmdk-api-gpu` в `docker-compose.yml` и убедитесь что установлен NVIDIA Container Toolkit.

---

## Часть 6: Полезные команды

```bash
# Запуск
docker-compose up -d

# Логи
docker-compose logs -f

# Проверить здоровье
curl http://localhost:8080/health

# Получить метрики (Prometheus format)
curl http://localhost:8080/v1/metrics

# Создать бэкап
curl -X POST http://localhost:8080/v1/backup/create -H "Content-Type: application/json" -d '{"name":"my_backup"}'

# Получить аналитику
curl http://localhost:8080/v1/analytics

# Экспорт в Markdown
curl http://localhost:8080/v1/export?format=markdown

# Отправить feedback
curl -X POST http://localhost:8080/v1/feedback -H "Content-Type: application/json" -d '{"query":"What do I drink?","quality":0.9}'

# Остановка
docker-compose down

# Остановка с удалением данных
docker-compose down -v
```

---

*Последнее обновление: Апрель 2026, RTMDK v8.0*
