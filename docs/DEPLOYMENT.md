# RTMDK — Версии для Развёртывания

## Сравнение версий

| Функция | Home (`legacy/rtmdk_server.py`) | Production (`python -m rtmdk`) |
|---------|:---:|:---:|
| OpenAI API (/v1/chat/completions) | ✅ | ✅ |
| Embeddings (/v1/embeddings) | ✅ | ✅ |
| Models (/v1/models) | ✅ | ✅ |
| Dashboard UI (/dashboard) | ✅ | ✅ |
| UX Endpoints (/api/*) | ✅ | ✅ |
| Memory (/v1/memory/*) | ✅ | ✅ |
| Security (API Key) | ✅ | ✅ |
| Auto-save | ✅ | ✅ |
| **SillyTavern endpoints** | ✅ | ❌ |
| **ST Proxy** | ✅ (отдельный процесс) | ❌ |
| **Размер** | ~1100 строк | ~350 строк |

## Какую версию выбрать?

### Home Version (`legacy/rtmdk_server.py`)
**Используйте когда:**
- Разрабатываете и тестируете
- Нужен SillyTavern integration
- Работаете с ролевыми играми
- Нужен полный набор endpoints

```bash
python legacy/rtmdk_server.py

# С пресетом
RTMDK_PRESET=local python legacy/rtmdk_server.py

# С параметрами
RTMDK_PRESET=local RTMDK_LATENT_DIM=128 python legacy/rtmdk_server.py
```

### Production Version (`python -m rtmdk`)
**Используйте когда:**
- Разворачиваете на сервере
- API для внешних приложений
- IDE интеграция (Cursor, Continue)
- Веб-приложения
- Чистый OpenAI-compatible API

```bash
python -m rtmdk
python start_production.py
python start_production.py --port 9090 --api-key mysecret
python start_production.py --no-auth  # Без авторизации
```

## Endpoints Production версии

### OpenAI-совместимые
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/v1/chat/completions` | POST | Чат с памятью |
| `/v1/embeddings` | POST | Эмбеддинги текста |
| `/v1/models` | GET | Список моделей |

### Управление
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Проверка здоровья |
| `/health/deep` | GET | Детальный health check |
| `/metrics` | GET | Prometheus metrics |
| `/v1/models` | GET | Список моделей |
| `/v1/admin/config` | POST | Runtime конфигурация |

### Память
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/v1/memory/nodes` | POST/GET | CRUD нод |
| `/v1/memory/nodes/{node_id}` | GET/PUT/DELETE | CRUD ноды |
| `/v1/memory/batch_ingest` | POST | Batch ingest |
| `/v1/memory/export` | GET | Экспорт |
| `/v1/memory/import` | POST | Импорт |
| `/v1/memory/query` | POST | Query |
| `/v1/memory/query_pipeline` | POST | Pipeline query |

### Admin
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/v1/admin/api-keys` | POST/GET | API keys |
| `/v1/admin/audit-log` | GET | Audit log |
| `/v1/admin/retention` | GET | Retention stats |
| `/v1/admin/tenants` | GET | Tenants |
| `/v1/admin/cache` | GET | Cache status |
| `/v1/admin/encryption` | GET | Encryption status |
| `/v1/admin/telemetry` | GET | Telemetry status |

### SOT & Replication
| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/v1/sot/*` | GET/POST | SOT management |
| `/v1/replication/*` | POST/GET | Replication |
| `/v1/webhooks` | POST/GET/DELETE | Webhooks |

## Безопасность Production

- **API Key**: `Authorization: Bearer <key>` или `x-api-key: <key>`
- **File permissions**: `0o600` на `memory.json`
- **Payload limit**: 1MB по умолчанию
- **CORS**: настраиваемый через `RTMDK_ALLOWED_ORIGINS`

## Переменные окружения

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `RTMDK_PORT` | `8080` | Порт сервера |
| `RTMDK_HOST` | `0.0.0.0` | Хост |
| `RTMDK_API_KEY` | `rtmdk-local` | API ключ |
| `RTMDK_ENABLE_API_AUTH` | `true` | Включить авторизацию |
| `RTMDK_LM_STUDIO_URL` | `http://localhost:12345/v1` | URL LM Studio |
| `RTMDK_MEMORY_FILE` | `~/.rtmdk/memory.json` | Файл памяти |
| `RTMDK_EMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5-GGUF` | Модель эмбеддера |
| `RTMDK_AUTO_SAVE_INTERVAL` | `60` | Интервал авто-сохранения (сек) |
| `RTMDK_MAX_PAYLOAD_SIZE` | `1048576` | Макс. размер запроса (1MB) |
| `RTMDK_ALLOWED_ORIGINS` | `*` | Разрешённые CORS origins |

## Docker Production

```bash
# Только сервер (без ST proxy)
docker-compose up rtmdk-server

# С кастомным API ключом
RTMDK_API_KEY=mysecret docker-compose up rtmdk-server
```
