# RTMDK — Resonance-Topological Memory v8.0

> Долгосрочная память для LLM на основе резонансной топологии и диалектической консолидации

---

## 📦 Две версии проекта

Проект разделён на **две конфигурации** для разных сценариев использования:

| | 🏭 Production | 🏠 Home / SillyTavern |
|---|---|---|
| **Назначение** | API-сервер для продакшена, микросервисы | Домашнее использование с SillyTavern |
| **Порты** | `8080` (один) | `8080` + `5000` (proxy) |
| **SillyTavern** | ❌ Не включён | ✅ Полная поддержка |
| **Dashboard** | ✅ Веб-UI | ✅ Веб-UI |
| **Dockerfile** | `Dockerfile` | `Dockerfile.home` |
| **Compose** | `docker-compose.prod.yml` | `docker-compose.home.yml` |
| **Зависимости** | Минимальные | Полные |
| **Запуск** | `start_production.bat` / `start_production.py` | `start_sillytavern.bat` / `rtmdk_sillytavern_launcher.py` |

---

## 🚀 Быстрый старт

### 🏭 Production (без SillyTavern)

```bash
pip install -r requirements-prod.txt
python start_production.py
# или: start_production.bat
```

### 🏠 Home / SillyTavern

```bash
pip install -r requirements-home.txt
python rtmdk_sillytavern_launcher.py
# или: start_sillytavern.bat
```

### 🔧 Monolith (один процесс, ST endpoints встроены)

```bash
python rtmdk_server.py
```

---

## ⚙️ Конфигурация через пресеты

RTMDK использует **единственный источник конфигурации** с 8 готовыми пресетами:

```python
from rtmdk import RTMDKConfig

config = RTMDKConfig.local()       # Персональный ассистент (~16MB)
config = RTMDKConfig.production()  # Продакшен сервер (~50MB)
config = RTMDKConfig.research()    # Максимальная точность (~200MB)
config = RTMDKConfig.enterprise()  # 100K+ узлов, distributed
config = RTMDKConfig.agent()       # Автономный агент
config = RTMDKConfig.legal()       # Юриспруденция (Z3 prover)
config = RTMDKConfig.medical()     # Медицина (Z3 + trust)
config = RTMDKConfig.streaming()   # High-throughput (~3ms)
```

### Переопределение через переменные окружения

Любой параметр можно переменить через `RTMDK_*` env var:

```bash
# Выбрать пресет
RTMDK_PRESET=production python rtmdk_server.py

# Переопределить отдельные параметры
RTMDK_LATENT_DIM=128 RTMDK_TOP_K=10 python rtmdk_server.py

# Комбинация
RTMDK_PRESET=research RTMDK_DECAY_RATE=0.9995 python rtmdk_server.py
```

**Поддерживаемые env vars:** `RTMDK_PRESET`, `RTMDK_LATENT_DIM`, `RTMDK_DECAY_RATE`,
`RTMDK_TENSION_THRESHOLD`, `RTMDK_TOP_K`, `RTMDK_BANDWIDTH`, `RTMDK_PHASE_COUPLING`,
`RTMDK_USE_HNSW`, `RTMDK_HNSW_M`, `RTMDK_LEARN_PROJECTION`, `RTMDK_CROSS_MODAL`,
`RTMDK_CAUSAL_TOPOLOGICAL`, `RTMDK_META_ADAPTIVE`, `RTMDK_SELF_HEALING`,
`RTMDK_ENABLE_ENGRAMS`, `RTMDK_OFFLINE_DREAMING`, `RTMDK_CAUSAL_TRAVERSAL`,
`RTMDK_SSM_DYNAMICS`, `RTMDK_SPARSE_ROUTING`, `RTMDK_NUM_SHARDS`, `RTMDK_GOAL_TRACKING`,
`RTMDK_RL_FEEDBACK`, `RTMDK_SECURITY_ENABLED`, `RTMDK_SWARM_MEMORY`,
`RTMDK_SYMBOLIC_OVERLAY`, `RTMDK_SAFETY_CERTIFIER`, `RTMDK_ROLE_SHARDING`,
`RTMDK_CONTEXT_FORMAT`, `RTMDK_LOG_LEVEL` и другие.

### Изменение конфигурации через API

```bash
# Изменить пресет (требует перезапуска)
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"RTMDK_PRESET": "production"}'
# Ответ: {"status":"ok", "needs_restart": true, "updates": ["RTMDK_PRESET"]}

# Изменить гиперпараметр (сохраняется в .env, требует перезапуска)
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"RTMDK_LATENT_DIM": "128", "RTMDK_TOP_K": "10"}'

# Изменить модель эмбеддера (применяется сразу)
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"RTMDK_EMBED_MODEL": "text-embedding-3-small"}'
```

---

## 🔌 SillyTavern подключение

| Режим | API Type | Base URL | API Key |
|-------|----------|----------|---------|
| **Proxy** (рекомендуется) | OpenAI | `http://127.0.0.1:5000/v1` | любой |
| **Monolith** | OpenAI | `http://127.0.0.1:8080/v1` | `rtmdk-local` |
| **Monolith** (Text Completion) | Text Completion | `http://127.0.0.1:8080` | — |

Подробнее: [SILLYTAVERN_CONNECTION_GUIDE.md](SILLYTAVERN_CONNECTION_GUIDE.md)

---

## 📚 Документация

| Что нужно | Документ |
|-----------|----------|
| **Главный индекс** | [docs/README.md](docs/README.md) |
| **API справка** | [docs/01_API_REFERENCE.md](docs/01_API_REFERENCE.md) |
| **Production 100K+** | [docs/02_PRODUCTION_GUIDE.md](docs/02_PRODUCTION_GUIDE.md) |
| **Локальный запуск** | [docs/03_LOCAL_SETUP.md](docs/03_LOCAL_SETUP.md) |
| **Docker + Silly Tavern** | [docs/04_DOCKER_SETUP.md](docs/04_DOCKER_SETUP.md) |
| **Тонкая настройка** | [docs/05_FINE_TUNING.md](docs/05_FINE_TUNING.md) |
| **Научная статья** | [docs/06_SCIENTIFIC_ARTICLE.md](docs/06_SCIENTIFIC_ARTICLE.md) |
| **Архитектура** | [docs/08_ARCHITECTURE.md](docs/08_ARCHITECTURE.md) |

## 📊 Результаты

| Метрика | Значение | vs RAG |
|---------|:---:|---|
| **Recall@1** | **95.2%** | +15-35% |
| **Recall@5** | **98.2%** | +13-28% |
| **Latency P95** | 132ms | В 3-15x быстрее |
| **RAM (1K узлов)** | 16 MB | В 3-12x экономнее |

## 🏗️ Архитектура

```
RTMDK v8.0 (25,000+ строк, 75+ файлов, 105+ API)
├── Core: Резонанс, консолидация, HNSW, BM25 (Phase 1-14)
├── Production: Version Control, Attention Tokens (Phase 15)
├── Safety: Symbolic Overlay, UMP, Safety Certifier (Phase 16)
├── Scale: Role Sharding, Swarm Memory (Phase 17)
├── Engrams: Pattern completion, engram decay (Phase 18)
└── Advanced: Offline Dreaming, Causal Traversal, SSM/Mamba,
    Trust Consensus, Neuro-Symbolic Prover (Phase 19)
```

## 📦 Поддерживаемые API

| Провайдер | Переменная |
|-----------|-----------|
| LM Studio (локально, бесплатно) | `RTMDK_API_PROVIDER=lm_studio` |
| OpenRouter (унифицированный) | `RTMDK_API_PROVIDER=openrouter` |
| OpenAI (официальный) | `RTMDK_API_PROVIDER=openai` |
| Anthropic (официальный) | `RTMDK_API_PROVIDER=anthropic` |
| Custom (Groq, Together, LocalAI) | `RTMDK_API_PROVIDER=custom` |

---

*RTMDK v8.0 — Превосходит GraphRAG, Self-RAG и Advanced RAG по точности*
