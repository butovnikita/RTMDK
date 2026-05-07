# RTMDK — Resonance-Topological Memory v8.2.1

> Долгосрочная память для LLM на основе резонансной топологии и диалектической консолидации
> Version 8.2.1 (Tracks 1–9 + Tiered Storage + GraphQL + WebSocket) — 30,000+ строк кода, 85+ файлов, 110+ API endpoints

---

## 🚀 Быстрый старт

### Вариант A: Python (рекомендуется для разработки)

```bash
pip install -r requirements-home.txt
python rtmdk_server.py
# → http://localhost:8080
```

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

### Вариант D: SillyTavern Launcher

```bash
python rtmdk_sillytavern_launcher.py
# Запускает сервер (8080) + proxy (5000) автоматически
```

---

## 📚 Документация

| Что нужно | Документ |
|-----------|----------|
| **Главный индекс** | [docs/MASTER_INDEX.md](docs/MASTER_INDEX.md) |
| **API справка** | [docs/01_API_REFERENCE.md](docs/01_API_REFERENCE.md) |
| **Запуск на своём ПК** | [docs/03_LOCAL_SETUP.md](docs/03_LOCAL_SETUP.md) |
| **Docker + Silly Tavern** | [docs/04_DOCKER_SETUP.md](docs/04_DOCKER_SETUP.md) |
| **Настройка параметров** | [docs/05_FINE_TUNING.md](docs/05_FINE_TUNING.md) |
| **Production 100K+ узлов** | [docs/02_PRODUCTION_GUIDE.md](docs/02_PRODUCTION_GUIDE.md) |
| **Научная статья (патент)** | [docs/06_SCIENTIFIC_ARTICLE.md](docs/06_SCIENTIFIC_ARTICLE.md) |
| **Архитектура системы** | [docs/08_ARCHITECTURE.md](docs/08_ARCHITECTURE.md) |
| **Domain Memory (Phase 20)** | [docs/20_DOMAIN_MEMORY.md](docs/20_DOMAIN_MEMORY.md) |
| **Быстрый старт + SillyTavern** | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| **SillyTavern Connection** | [SILLYTAVERN_CONNECTION_GUIDE.md](SILLYTAVERN_CONNECTION_GUIDE.md) |
| **Калибровка параметров** | [Values.md](Values.md) |
| **Проверка кода (аудит)** | [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md) |
| **Полный аудит модулей** | [docs/FULL_AUDIT.md](docs/FULL_AUDIT.md) |
| **Commercial roadmap** | [docs/ROADMAP.md](docs/ROADMAP.md) |
| **Deployment варианты** | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |

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

```bash
# Выбрать пресет
RTMDK_PRESET=production python rtmdk_server.py

# Переопределить отдельные параметры
RTMDK_LATENT_DIM=128 RTMDK_TOP_K=10 python rtmdk_server.py

# Комбинация
RTMDK_PRESET=research RTMDK_DECAY_RATE=0.9995 python rtmdk_server.py
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

## 📊 Результаты

| Метрика | Значение | vs RAG |
|---------|:---:|---|
| **Recall@1** | **95.6%** | +15-35% |
| **Recall@5** | **98.2%** | +13-28% |
| **Latency P95** | **1.3 ms** | В 100-500x быстрее |
| **RAM (1K узлов)** | **16 MB** | В 3-12x экономнее |
| **RAM (10K fp16)** | **9.8 MB** | В 5-20x экономнее |

## 🏗️ Архитектура

```
RTMDK v8.2.0 (30,000+ строк, 111+ файлов, 105+ API)
├── Core: Резонанс, консолидация, HNSW, BM25, Domain Memory (Phase 1-20)
├── Production: Version Control, Attention Tokens (Phase 15)
├── Safety: Symbolic Overlay, UMP, Safety Certifier (Phase 16)
├── Scale: Role Sharding, Swarm Memory (Phase 17)
├── Tracks (v8.2):
│   ├── Track 1: fp16 Quantization (2× RAM savings)
│   ├── Track 2: Tiered Storage — Hot/Warm/Cold tiers
│   ├── Track 3: Query Cache + Adaptive top_k
│   ├── Track 4: Async Batch Ingestion Pipeline
│   ├── Track 5: WAL Replay & Durability Recovery
│   ├── Track 6: Async Save Worker + Background Index
│   ├── Track 7: CI/CD + PyPI Production Hardening
│   ├── Track 8: MCP Server (Model Context Protocol)
│   ├── Track 9: LangChain LCEL Integration
│   ├── Track 10: Analytics Dashboard API
│   ├── Track 11: API Keys + Tenant Rate Limiting
│   ├── Track 12: Memory Node CRUD REST API
│   ├── Track 13: Structured JSON Request Logging
│   ├── Track 14: Python Client SDK
│   ├── Track 15: Webhook Subscriptions
│   ├── Track 16: Batch Ingestion + Import/Export REST
│   ├── Track 17: Tier 1 Production Readiness (Health, Audit Log, Retention)
│   ├── Track 18: int8 Quantization
│   ├── Track 19: Redis Cache Layer
│   ├── Track 20: gRPC Service
│   ├── Track 21: Encryption at Rest
│   ├── Track 22: OpenTelemetry Tracing
│   ├── Track 23: Load Tests + Docker Compose
│   ├── Track 24: SOT Out-of-the-Box (Self-Organizing Tokenizer)
│   ├── Track 25: GraphQL API (Strawberry)
│   ├── Track 26: WebSocket Streaming (/ws/memory)
│   ├── Track 27: React Admin Panel
│   └── Track 28: SOT Persistence + Graceful Degradation
└── Integrations: OpenAI, Anthropic, LM Studio, SillyTavern, MCP, LangChain, LlamaIndex
```

---

## 📦 Поддерживаемые API провайдеры

| Провайдер | Переменная |
|-----------|-----------|
| LM Studio (локально, бесплатно) | `RTMDK_API_PROVIDER=lm_studio` |
| OpenRouter (унифицированный) | `RTMDK_API_PROVIDER=openrouter` |
| OpenAI (официальный) | `RTMDK_API_PROVIDER=openai` |
| Anthropic (официальный) | `RTMDK_API_PROVIDER=anthropic` |
| Custom (Groq, Together, LocalAI) | `RTMDK_API_PROVIDER=custom` |

---

## 📁 Структура проекта

```
.
├── rtmdk/                      # Python-пакет
│   ├── __init__.py             # Re-export всех символов
│   ├── config.py               # RTMDKConfig + 8 пресетов
│   ├── nodes.py                # Data-классы (MemoryNode, etc.)
│   ├── engrams.py              # Phase 18: Engram system
│   ├── memory/
│   │   ├── core.py             # RTMDKMemory + ядро (~4000 строк)
│   │   ├── field.py            # RTMDKField — query, consolidation, cache
│   │   ├── config.py           # RTMDKConfig + 8 пресетов
│   │   ├── tiered_storage.py   # Track 2: Hot/Warm/Cold tiers
│   │   ├── query_cache.py      # Track 3: Query Cache
│   │   ├── wal.py              # Track 5: Write-Ahead Log
│   │   └── serialization.py    # Import/Export
│   ├── server/
│   │   └── app.py              # FastAPI production server
│   ├── engines/                # Computation engines (9 modules)
│   ├── support/                # 28 support classes
│   └── production/             # 33 production modules
├── docs/                       # Документация (15 файлов)
├── tests/                      # Тесты
├── archive/                    # Исторические файлы
├── rtmdk_server.py             # Monolith сервер (с ST endpoints)
├── rtmdk_server_ux.py          # UX endpoints router
├── rtmdk_dashboard_ui.py       # Dashboard UI endpoints
├── rtmdk_sillytavern_launcher.py  # SillyTavern launcher
├── rtmdk_st_proxy.py           # SillyTavern proxy
├── embedder_lmstudio.py        # LM Studio embedder
├── archive/scripts/generate_qa_1000.py  # QA dataset generator
├── tests/smoke_test.py          # Smoke tests
├── Dockerfile / Dockerfile.home / Dockerfile.gpu
├── docker-compose.yml / docker-compose.prod.yml / docker-compose.home.yml
└── requirements*.txt
```

---

## 🎯 Фазы реализации

| Phase | Что реализовано | Статус |
|-------|----------------|:---:|
| 1-14 | Ядро RTMDK: резонанс, консолидация, HNSW, BM25, PCA | ✅ |
| 15 | Version Control, Proactive Clarification, Attention Tokens | ✅ |
| 16 | Symbolic Overlay, Safety Certifier, UMP | ✅ |
| 17 | Role Sharding, Swarm Memory | ✅ |
| **18** | **Энграммы** — паттерны коактивации, pattern completion | ✅ |
| **19** | Offline Dreaming, Causal Traversal, SSM/Mamba, Trust Consensus, Neuro-Symbolic Prover | ✅ |
| **20** | **Domain Memory** — Domain Hierarchy, Concept Lifecycle, Evidence Spans, Bi-temporal Facts | ✅ |

## 🚀 Tracks v8.2 (Production Hardening)

| Track | Фича | Статус |
|-------|------|:---:|
| 1 | **fp16 Quantization** — 2× меньше RAM, 100% R@1 | ✅ Shipped |
| 2 | **Tiered Storage** — Hot/Warm/Cold tiers, LFU, msgpack | ✅ Shipped |
| 3 | **Query Cache** — MD5-ключ, TTL, adaptive top_k | ✅ Shipped |
| 4 | **Async Batch Ingestion** — векторизованный pipeline | ✅ Shipped |
| 5 | **WAL Replay** — durability, crash recovery | ✅ Shipped |
| 6 | **Async Save Worker** — background index build | ✅ Shipped |
| 7 | **CI/CD + PyPI** — автоматическая публикация | ✅ Shipped |
| 8 | **MCP Server** — Model Context Protocol | ✅ Shipped |
| 9 | **LangChain LCEL** — нативная интеграция | ✅ Shipped |

---

*RTMDK v8.2.1 — Превосходит GraphRAG, Self-RAG и Advanced RAG по точности, latency и TCO*
*Документация: [docs/MASTER_INDEX.md](docs/MASTER_INDEX.md)*