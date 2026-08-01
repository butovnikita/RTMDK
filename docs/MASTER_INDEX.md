# RTMDK v8.3.1 — Полная документация

> Resonance-Topological Memory для LLM — версия 8.3.1 (Phase 23)
> Репозиторий: 440+ файлов, 74,000+ строк кода, 49 API endpoints

---

## Quick Start (Быстрый старт)

```bash
# Вариант 1: Python (рекомендуется)
pip install -r requirements-home.txt
python legacy/rtmdk_server.py

# Вариант 2: Docker
docker-compose -f docker-compose.home.yml up -d

# Вариант 3: SillyTavern
python legacy/rtmdk_sillytavern_launcher.py
```

---

## Структура документации

### Перед началом работы
| Документ | Описание |
|----------|----------|
| **[README.md](../README.md)** | Обзор проекта, быстрый старт, сравнение с RAG |
| **[MASTER_INDEX.md](./MASTER_INDEX.md)** | Этот файл — навигация по всей документации |
| **[Values.md](../Values.md)** | Калибровка гиперпараметров — все параметры с рекомендациями |

### Установка и настройка
| Документ | Описание |
|----------|----------|
| **[03_LOCAL_SETUP.md](./03_LOCAL_SETUP.md)** | Локальный запуск на своём ПК — Python, Docker, LM Studio, React Admin Panel |
| **[04_DOCKER_SETUP.md](./04_DOCKER_SETUP.md)** | Docker + SillyTavern — все docker-compose варианты |
| **[ST_PROXY_SETUP.md](./ST_PROXY_SETUP.md)** | RTMDK Proxy для SillyTavern — архитектура и настройка |
| **[SILLYTAVERN_CONNECTION_GUIDE.md](../SILLYTAVERN_CONNECTION_GUIDE.md)** | Подключение SillyTavern — 3 варианта (Monolith/Proxy/Text Completion) |
| **[QUICKSTART.md](./QUICKSTART.md)** | 5-минутный quick start с SillyTavern |

### Конфигурация
| Документ | Описание |
|----------|----------|
| **[05_FINE_TUNING.md](./05_FINE_TUNING.md)** | Полное руководство по настройке — 59 env vars, 8 профилей |
| **[DEPLOYMENT.md](./DEPLOYMENT.md)** | Две версии (Home/Production) — что выбрать и почему |

### Архитектура и API
| Документ | Описание |
|----------|----------|
| **[01_API_REFERENCE.md](./01_API_REFERENCE.md)** | Полный API reference — 30+ endpoints, все методы |
| **[08_ARCHITECTURE.md](./08_ARCHITECTURE.md)** | Архитектура системы — все фазы, модули, поток данных |
| **[20_DOMAIN_MEMORY.md](./20_DOMAIN_MEMORY.md)** | Phase 20: Domain Hierarchy, Concept Lifecycle, Evidence Spans |

### Специализированные руководства
| Документ | Описание |
|----------|----------|
| **[02_PRODUCTION_GUIDE.md](./02_PRODUCTION_GUIDE.md)** | Scaling от 100K до 10M узлов, distributed architecture |
| **[06_SCIENTIFIC_ARTICLE.md](./06_SCIENTIFIC_ARTICLE.md)** | Научная статья для патента — алгоритмы, доказательства, benchmarks |
| **[07_DIALOGUE_EXPORT.md](./07_DIALOGUE_EXPORT.md)** | История разработки, version changelog, ключевые решения |

### Аудит и качество
| Документ | Описание |
|----------|----------|
| **[CODE_REVIEW.md](./CODE_REVIEW.md)** | Code audit — 241 баг исправлен, security review, рекомендации |
| **[FULL_AUDIT.md](./FULL_AUDIT.md)** | Полный аудит ядра и модулей — дублирование, орфанные модули |
| **[ROADMAP.md](./ROADMAP.md)** | Commercial roadmap — защита, технический план, маркетинг, финансы |

---

## Ключевые концепции

### Единый источник конфигурации
RTMDK использует **единственный** `RTMDKConfig` dataclass с 8 пресетами:

```python
from rtmdk import RTMDKConfig

config = RTMDKConfig.local()       # ~16MB RAM, до 10K узлов
config = RTMDKConfig.production()  # ~50MB RAM, до 100K узлов
config = RTMDKConfig.research()    # ~200MB RAM, без лимита
config = RTMDKConfig.enterprise()  # distributed, 500K+ узлов
config = RTMDKConfig.agent()       # autonomous agent
config = RTMDKConfig.legal()       # Z3 prover, legal domain
config = RTMDKConfig.medical()     # Z3 + trust, medical domain
config = RTMDKConfig.streaming()   # ~3ms latency, high-throughput
```

### Фазы реализации

| Phase | Что реализовано | Статус |
|-------|----------------|:---:|
| 1-14 | Ядро RTMDK: resonance, consolidation, HNSW, BM25, PCA | ✅ |
| 15 | Version Control, Proactive Clarification, Attention Tokens | ✅ |
| 16 | Symbolic Overlay, Safety Certifier, UMP | ✅ |
| 17 | Role Sharding, Swarm Memory | ✅ |
| **18** | **Энграммы** — pattern completion, engram decay | ✅ |
| **19** | Offline Dreaming, Causal Traversal, SSM/Mamba, Trust Consensus, Neuro-Symbolic Prover | ✅ |
| **20** | **Domain Memory** — hierarchy, concept lifecycle, evidence spans | ✅ |

### Метрики производительности

| Метрика | RTMDK | vs RAG |
|---------|:-----:|:------:|
| **Recall@1** | 95.2% | +15-35% |
| **Recall@5** | 98.2% | +13-28% |
| **Latency P95** | 132ms (baseline) / 8ms (engrams) | в 3-15x быстрее |
| **RAM (1K узлов)** | 16-18 MB | в 3-12x экономнее |

---

## Поддерживаемые API провайдеры

| Провайдер | Переменная | Примечание |
|-----------|-----------|------------|
| **LM Studio** (локально, бесплатно) | `RTMDK_API_PROVIDER=lm_studio` | Рекомендуется для разработки |
| **OpenRouter** | `RTMDK_API_PROVIDER=openrouter` | Унифицированный доступ |
| **OpenAI** | `RTMDK_API_PROVIDER=openai` | Официальный API |
| **Anthropic** | `RTMDK_API_PROVIDER=anthropic` | Claude models |
| **Custom** | `RTMDK_API_PROVIDER=custom` | Groq, Together, LocalAI |

---

## Развёртывание

### Две версии проекта

| Функция | Home (`legacy/rtmdk_server.py`) | Production (`python -m rtmdk`) |
|---------|:---:|:---:|
| OpenAI API | ✅ | ✅ |
| Dashboard UI | ✅ | ✅ |
| UX Endpoints | ✅ | ✅ |
| SillyTavern endpoints | ✅ | ❌ |
| ST Proxy | ✅ (отдельный процесс) | ❌ |
| Размер | ~1100 строк | ~350 строк |

Подробнее: [DEPLOYMENT.md](./DEPLOYMENT.md)

### Docker варианты

| Файл | Назначение | Размер |
|------|-----------|--------|
| `Dockerfile` | Production (CPU) | ~200MB |
| `Dockerfile.home` | Home + SillyTavern | ~400MB |
| `Dockerfile.gpu` | GPU (CUDA 12.1) | ~4GB |

---

## Структура кода

```
rtmdk/                              # Python-пакет
├── __init__.py                     # Re-export всех символов
├── config.py                       # RTMDKConfig + 8 пресетов
├── nodes.py                        # Data-классы (MemoryNode, etc.)
├── engrams.py                      # Phase 18: Engram system
├── memory/
│   ├── core.py                     # ЕДИНСТВЕННЫЙ RTMDKConfig + RTMDKMemory (~6773 строк)
│   └── serialization.py            # Import/Export
├── server/
│   └── app.py                      # FastAPI production server
├── engines/                        # Computation engines
│   ├── causal.py                   # CausalInferenceEngine
│   ├── causal_traversal.py         # Phase 19: BFS traversal
│   ├── counterfactual.py           # ScenarioPlanner
│   ├── neural_ode.py               # NeuralODEDynamics
│   ├── neuro_symbolic_prover.py    # Z3/Prolog integration
│   ├── predictive.py               # PredictiveCodingModel
│   ├── privacy.py                  # DifferentialPrivacy
│   ├── ssm_dynamics.py             # State Space Models (Mamba)
│   └── trust_consensus.py         # DAG-based trust
├── support/                        # 28 support classes
│   ├── bm25.py, hnsw.py           # Indexing
│   ├── meta_controller.py         # Optuna optimization
│   ├── healer.py                  # TopologyHealer
│   ├── kuramoto.py               # FederatedRTMDK
│   └── ...
└── production/                     # 33 production modules
    ├── offline_dreamer.py         # Background cycles
    ├── query_cache.py            # LRU cache
    └── ...
```

---

## Версии файлов

| Дата | Версия | Изменения |
|------|--------|----------|
| Апрель 2026 | 8.1 | Phase 20 (Domain Memory), bug fixes |
| Апрель 2026 | 8.0 | Unified config, 8 presets, 59 env vars |
| Март 2026 | 7.x | Previous stable version |

---

*Документация актуальна для RTMDK v8.3*
*Последнее обновление: Апрель 2026*