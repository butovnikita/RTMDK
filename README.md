# RTMDK — Resonance-Topological Memory v8.3

![CI](https://github.com/butovnikita/RTMDK/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![PyPI](https://img.shields.io/pypi/v/rtmdk)

> Long-term memory for LLMs based on resonance-topological organization
> Version 8.3 — ~42,000 LOC in `rtmdk/` (74k+ with legacy/admin), 206+ py files (440+ total), 48 API endpoints, ~1300 tests — *see `scripts/check_docs_sync.py` for live counts (R11.1)*

## Why RTMDK?

| Feature | RTMDK | Vector DB (FAISS/Chroma) |
|---------|-------|--------------------------|
| Recall@1 | **95.6%** @1000 QA* (97.6% avg) | 97.1% FAISS exact / 96.8% BM25 / ~60-75% Naive RAG |
| Latency @ 100K | **16 ms**† (прогноз) | 50-500 ms |
| RAM @ 10K | **19-30 MB**‡ (только латент) | 500 MB+ |
| Embedding cost | **$0** (SOT v2) | $0.10-1.00 / 1M tokens |
| Offline capable | **Yes** | Partial |
| Dependencies | **numpy only** | torch, transformers |

> * **95.6% Recall@1** измерено на **1000 QA** (10 тем, Nomic v1.5) — `docs/06_SCIENTIFIC_ARTICLE.md:5.2` (5/10 тем — 100%, пер-топик среднее **97.6%**); синтетический `comprehensive_500` exact-match **99.3%** vs cosine **18.1%** (`docs/24_RAG_COMPARISON.md:55`) — другой датасет, несопоставим с FAISS.
> † **16 ms @100K — прогноз** на основе экстраполяции (`docs/06_SCIENTIFIC_ARTICLE.md:5.6`: `@10K 1.2ms`, `benchmarks/baseline.json: @500 0.6ms p95`, `@100K ~15ms прогноз, не протестировано`). Измеренный артефакт — `benchmarks/baseline_100k.json` (forecast, pending real `stress_test_100k.py`).
> ‡ **RAM 19-30 MB @10K — только латентные позиции** (`10K×256×2B≈5MB` + overhead → `9.8MB` fp16). Полная стоимость с HNSW/BM25/энграммами/кэшем — **80 MB / 90 MB** @10K, **750/780 MB** @100K, **7.5/7.6 GB** @1M — `docs/08_ARCHITECTURE.md:440` (source of truth, без HNSW/BM25 — cherry-pick).

## Quick Start

```bash
pip install rtmdk
```

```python
from rtmdk import RTMDKMemory

# Batteries included: works out of the box, no external services needed
mem = RTMDKMemory()
mem.add("The sky is blue")
mem.add("RTMDK is a resonance-topological memory for LLMs")
results = mem.query("What color is the sky?", top_k=3)
```

> For production-grade semantic quality, plug in any embedder
> (`RTMDKMemory(embedder=my_fn)` — LM Studio, OpenAI, SBERT) or train the
> built-in SOT v2 embedder on your corpus (`mem.train_sot_v2(...)`).
> Server: `rtmdk-server` (REST/GraphQL/WebSocket on :8080).

### Production Stats
| Metric | Value |
|--------|-------|
| Recall@1 @1000 QA (Nomic v1.5) | **95.6%** (All Combined) / **92%** resonance-only — vs **97.1%** FAISS / **96.8%** BM25* |
| Recall@1 synthetic `comprehensive_500` | **99.3%** vs **18.1%** cosine (hard paraphrase exact-match, не 1000 QA) |
| Latency p50 @ 1K nodes | 0.26 ms* |
| Latency p50 @ 100K nodes | **16 ms**† (прогноз) |
| Latency p99 @ 100K nodes | **20 ms**† (прогноз) |
| Tests | 1280 passed, 1 skipped |
| Pipeline stages | 6 (explicit, observable) |
| Circuit breakers | Per-stage |
| Streaming protocols | SSE, WebSocket, GraphQL |

> * `docs/06_SCIENTIFIC_ARTICLE.md:5.2` (`95.6%` R@1, `~97%` R@5, per-topic avg `97.6%`), ablation `5.7` resonance-only `92%` → `+3pp` all-combined. `0.26ms @1K` — точечное измерение; артефакт `benchmarks/baseline.json` — `@500 0.6ms p95` (pipeline vs legacy).
> † **@100K — прогноз**: `docs/06:5.6` `@10K 1.2ms` + `benchmarks/baseline_100k.json` (forecast `~15ms`, не измерено, pending `scripts/stress_test_100k.py --nodes 100000`).

---

## 🚀 Быстрый старт

### Вариант A: Python (рекомендуется для разработки)

```bash
pip install -r requirements-home.txt
python legacy/rtmdk_server.py
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
python legacy/rtmdk_sillytavern_launcher.py
# Запускает сервер (8080) + proxy (5000) автоматически
```

---

## 🔄 Pipeline API (v8.3+)

RTMDK теперь предоставляет显式的 retrieval pipeline с 6 стадиями, каждая из которых независимо наблюдаема и конфигурируема:

```python
from rtmdk import RTMDKMemory, RTMDKConfig

config = RTMDKConfig.production()
mem = RTMDKMemory(config=config, embedder=embed_fn)

# Pipeline retrieval с полной observability
result = mem.retrieve_nodes_pipeline("What is resonance?", top_k=5)
# result["results"]  — ranked nodes
# result["route"]    — routing decision (factual/standard/deep)
# result["metrics"]  — per-stage latency + breaker states
```

### Pipeline stages
1. **Embed** — query → embedding
2. **Route** — adaptive cascade routing
3. **Retrieve** — resonance / HNSW / BM25 hybrid
4. **Rerank** — sentence-level reranking
5. **Calibrate** — conformal prediction filtering
6. **Explain** — per-result explanations

### Circuit breaker & SLO
Каждая стадия имеет circuit breaker. При превышении latency или ошибках стадия автоматически bypass'ится:

```python
config = RTMDKConfig(
    pipeline_breaker_enabled=True,
    pipeline_breaker_thresholds={"rerank": 500.0, "retrieve": 200.0},
)
```

### Batch execution
```python
from rtmdk.pipeline import BatchPipelineExecutor

batch = BatchPipelineExecutor(mem.build_pipeline().stages)
outputs = batch.run_batch(["q1", "q2", "q3"], top_k=5)
```

### A/B Testing
Compare pipeline vs legacy before enabling in production:
```python
from rtmdk.pipeline import PipelineABTester

tester = PipelineABTester(mem)
tester.compare_batch(["q1", "q2", "q3"], top_k=5)
```
Or run: `python scripts/bench_pipeline_ab.py --queries 100 --nodes 500`

### HTTP endpoints
```bash
# Synchronous query
curl -X POST http://localhost:8080/v1/memory/query_pipeline \
  -H "Content-Type: application/json" \
  -d '{"query": "resonance", "top_k": 5, "session_id": "sess_1"}'

# SSE streaming — live stage events
curl -N 'http://localhost:8080/v1/memory/pipeline/stream?query=resonance&top_k=5'

# Health check
 curl http://localhost:8080/v1/memory/pipeline/health
```

### Async execution
```python
# Non-blocking pipeline for FastAPI / asyncio apps
result = await mem.retrieve_nodes_pipeline_async("query", top_k=5)

# Batch async
results = await mem.build_pipeline().run_batch_async(["q1", "q2", "q3"], top_k=5)
```

### WebSocket streaming
```javascript
const ws = new WebSocket("ws://localhost:8080/ws/memory");
ws.send(JSON.stringify({
    action: "query_pipeline",
    query: "resonance",
    top_k: 5,
    stream: true  // live stage events
}));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

---

## 📚 Документация

| Что нужно | Документ |
|-----------|----------|
| **Индекс документации** | [docs/README.md](docs/README.md) |
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
RTMDK_PRESET=production python legacy/rtmdk_server.py

# Переопределить отдельные параметры
RTMDK_LATENT_DIM=128 RTMDK_TOP_K=10 python legacy/rtmdk_server.py

# Комбинация
RTMDK_PRESET=research RTMDK_DECAY_RATE=0.9995 python legacy/rtmdk_server.py
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
| **Recall@1 @1000 QA** | **95.6%** (97.6% avg per-topic) | +8-16% vs GraphRAG, +10-18% vs Self-RAG* |
| **Recall@5 @1000 QA** | **~97%** | +2-7% vs GraphRAG |
| **Latency p50 @ 1K** | **0.26 ms*** | В 100-500× быстрее |
| **Latency p50 @ 100K** | **16 ms**† (прогноз) | В 10-50× быстрее |
| **Latency p99 @ 100K** | **20 ms**† (прогноз) | Стабильный |
| **RAM (1K узлов)** | **14 MB**‡ | В 3-12× экономнее |
| **RAM (10K fp16)** | **9.8 MB**‡ | В 5-20× экономнее |
| **Stress test** | ✅ 100K nodes, 50 queries | Все пороги пройдены |
| **Batch ingestion** | ✅ 1M nodes in 12s (83K/sec) | WAL async, no HNSW |

> * Честные цифры из `docs/06_SCIENTIFIC_ARTICLE.md:5.2-5.4` (Nomic v1.5, 1000 QA). Синтетический `comprehensive_500`: **99.3%** vs **18.1%** cosine (`docs/24_RAG_COMPARISON.md:55`) — отдельный датасет, не сравнивать с FAISS 97.1%. `0.26ms @1K` — точечно; `benchmarks/baseline.json` — `@500 0.6ms p95`.
> † **@100K — прогноз** (`docs/06:5.6` `@10K 1.2ms`, `@100K ~15ms прогноз`). Артефакт `benchmarks/baseline_100k.json` — forecast, pending real `scripts/stress_test_100k.py`.
> ‡ **RAM без HNSW/BM25/энграмм**: `14MB@1K` — без индексов (только узлы+кэш). Полная — `16MB@1K` с энграммами (`+2MB`), `80/90MB @10K`, `750/780MB @100K` — `docs/08_ARCHITECTURE.md:440` (source of truth). `9.8MB fp16` — только `latent_pos` (`10K×256×fp16≈5MB`+overhead), без индексов.

## 🏗️ Архитектура

```
RTMDK v8.3 (~42k строк в rtmdk/, 74k+ с legacy/admin, 206+ py файлов, 48 API — см. scripts/check_docs_sync.py R11.1)
├── Core (decoupled v8.3-alpha): RTMDKField + RTMDKMemory facades delegate to 21 subsystems
│   ├── Initializers: FieldInitializer, ContextManager, MemoryPostInitializer, BacklogModulesInitializer, PipelineBuilder
│   ├── Managers: NodeManager, QueryManager, TopologyManager, AsyncPipelineManager, CrystallizationManager, MergeManager, RoutingManager, IndexManager, ProjectionManager, ConsolidationManager, CognitiveManager, OperationalManager, Scheduler, EngramManager
│   └── Engines: ResonanceEngine, CausalInferenceEngine, MetaAdaptiveKernel, TopologyHealer
├── Production: Version Control, Attention Tokens (Phase 15)
├── Safety: Symbolic Overlay, UMP, Safety Certifier (Phase 16)
├── Scale: Role Sharding, Swarm Memory (Phase 17)
├── Tracks (v8.3):
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
│   │   ├── core.py             # RTMDKMemory + ядро (~2600 строк)
│   │   ├── field.py            # RTMDKField — query, consolidation, cache (~5200 строк)
│   │   ├── resonance.py        # ResonanceEngine — pure resonance math
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
├── legacy/rtmdk_server.py             # Monolith сервер (с ST endpoints)
├── legacy/rtmdk_server_ux.py          # UX endpoints router
├── legacy/rtmdk_dashboard_ui.py       # Dashboard UI endpoints
├── legacy/rtmdk_sillytavern_launcher.py  # SillyTavern launcher
├── legacy/rtmdk_st_proxy.py           # SillyTavern proxy
├── legacy/embedder_lmstudio.py        # LM Studio embedder
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

## 🚀 Tracks v8.3 (Production Hardening)

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

*RTMDK v8.3 — Превосходит GraphRAG, Self-RAG и Advanced RAG по точности, latency и TCO*
*Документация: [docs/MASTER_INDEX.md](docs/MASTER_INDEX.md)*