# RTMDK Documentation

> Resonance-Topological Memory for LLMs — Version 8.1 (Phase 20)

## Навигация

**Главный индекс:** [MASTER_INDEX.md](./MASTER_INDEX.md) — полная навигация по всей документации

---

## Быстрый старт

| Что нужно | Документ |
|-----------|----------|
| **API справка** | [01_API_REFERENCE.md](01_API_REFERENCE.md) |
| **Запуск на своём ПК** | [03_LOCAL_SETUP.md](03_LOCAL_SETUP.md) |
| **Docker + Silly Tavern** | [04_DOCKER_SETUP.md](04_DOCKER_SETUP.md) |
| **Настройка параметров** | [05_FINE_TUNING.md](05_FINE_TUNING.md) |
| **Production 100K+ узлов** | [02_PRODUCTION_GUIDE.md](02_PRODUCTION_GUIDE.md) |
| **Научная статья (патент)** | [06_SCIENTIFIC_ARTICLE.md](06_SCIENTIFIC_ARTICLE.md) |
| **Архитектура системы** | [08_ARCHITECTURE.md](08_ARCHITECTURE.md) |
| **История разработки** | [07_DIALOGUE_EXPORT.md](07_DIALOGUE_EXPORT.md) |
| **Domain Memory (Phase 20)** | [20_DOMAIN_MEMORY.md](20_DOMAIN_MEMORY.md) |
| **Калибровка параметров** | [Values.md](../Values.md) |
| **Проверка кода (аудит)** | [CODE_REVIEW.md](CODE_REVIEW.md) |
| **Полный аудит модулей** | [FULL_AUDIT.md](FULL_AUDIT.md) |
| **Commercial roadmap** | [ROADMAP.md](ROADMAP.md) |
| **Deployment варианты** | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **Быстрый старт + SillyTavern** | [QUICKSTART.md](QUICKSTART.md) |
| **SillyTavern Connection** | [SILLYTAVERN_CONNECTION_GUIDE.md](../SILLYTAVERN_CONNECTION_GUIDE.md) |
| **ST Proxy Setup** | [ST_PROXY_SETUP.md](ST_PROXY_SETUP.md) |

---

## Конфигурация — единый источник

RTMDK использует **единственный** `RTMDKConfig` dataclass (`rtmdk/memory/core.py`) с 8 пресетами:

```python
from rtmdk import RTMDKConfig

config = RTMDKConfig.local()       # ~16MB, 10K nodes
config = RTMDKConfig.production()  # ~50MB, 100K nodes
config = RTMDKConfig.research()    # ~200MB, unlimited
config = RTMDKConfig.enterprise()  # distributed, 500K+
config = RTMDKConfig.agent()       # autonomous agent
config = RTMDKConfig.legal()       # Z3 prover
config = RTMDKConfig.medical()     # Z3 + trust
config = RTMDKConfig.streaming()   # ~3ms latency
```

### Env Var Overrides (59 переменных)

Любой параметр переопределяется через `RTMDK_*` env var:

```bash
RTMDK_PRESET=production RTMDK_LATENT_DIM=128 python rtmdk_server.py
```

Полный список: `RTMDK_LATENT_DIM`, `RTMDK_DECAY_RATE`, `RTMDK_TENSION_THRESHOLD`,
`RTMDK_TOP_K`, `RTMDK_BANDWIDTH`, `RTMDK_PHASE_COUPLING`, `RTMDK_USE_HNSW`,
`RTMDK_HNSW_M`, `RTMDK_LEARN_PROJECTION`, `RTMDK_CROSS_MODAL`, `RTMDK_CAUSAL_TOPOLOGICAL`,
`RTMDK_META_ADAPTIVE`, `RTMDK_SELF_HEALING`, `RTMDK_ENABLE_ENGRAMS`,
`RTMDK_OFFLINE_DREAMING`, `RTMDK_CAUSAL_TRAVERSAL`, `RTMDK_SSM_DYNAMICS`, и другие.

### Runtime Configuration

```bash
# Через API (сохраняется в .env)
curl -X POST http://localhost:8080/api/config \
  -d '{"RTMDK_PRESET": "production", "RTMDK_LATENT_DIM": "128"}'
# Ответ: {"status":"ok", "needs_restart": true, "restart_required_keys": [...]}
```

---

## Фазы реализации

| Фаза | Что реализовано | Статус |
|------|----------------|:---:|
| 1-14 | Ядро RTMDK: резонанс, консолидация, HNSW, BM25, PCA | ✅ |
| 15 | Version Control, Proactive Clarification, Attention Tokens | ✅ |
| 16 | Symbolic Overlay, Safety Certifier, UMP | ✅ |
| 17 | Role Sharding, Swarm Memory | ✅ |
| **18** | **Энграммы** — паттерны коактивации, pattern completion | ✅ |
| **19** | **Offline Dreaming**, Causal Traversal, SSM/Mamba, Trust Consensus, Neuro-Symbolic Prover | ✅ |
| 20+ | Active Inference, TPR, Adversarial Arena (research modes) | ✅ |

---

## Метрики производительности

| Метрика | Значение |
|---------|----------|
| **Recall@1** | 95.2% |
| **Recall@5** | 98.2% |
| **Latency P95** | 132ms (baseline) / 8ms (энграммы) |
| **RAM (1K узлов)** | 16-18 MB |
| **RAM (100K узлов)** | 780-800 MB |

---

## Ключевые улучшения vs RAG

| Система | Recall@1 | Latency P95 |
|---------|:---:|:---:|
| **RTMDK** | **95.2%** | 132ms |
| GraphRAG | 82-90% | 500ms-2s |
| Self-RAG | 80-88% | 300-800ms |
| Advanced RAG | 75-85% | 200-500ms |
| Naive RAG | 60-75% | 50-200ms |

---

## Структура пакета

```
rtmdk/
├── memory/core.py     # ЕДИНСТВЕННЫЙ RTMDKConfig + RTMDKMemory (~6200 строк)
├── config.py          # Пресеты (local/production/research/...) — импортирует из memory/core
├── nodes.py           # Data-классы узлов
├── engrams.py         # Phase 18: Энграммы
├── utils/             # Утилиты (modality, attention, formatting, hyperbolic)
├── engines/           # Движки
│   ├── causal_traversal.py    # Причинный обход
│   ├── ssm_dynamics.py        # State Space Models (Mamba)
│   ├── trust_consensus.py     # DAG доверия
│   └── neuro_symbolic_prover.py # Z3/Prolog
├── support/           # 28 классов поддержки
└── production/        # Production модули (33 файла)
    ├── offline_dreamer.py      # Фоновые циклы
    ├── query_cache.py          # LRU кэш запросов
    └── ...
```

### Импорт

```python
# Правильно — единый источник
from rtmdk import RTMDKConfig, RTMDKMemory
from rtmdk import ConsolidationMode, Backend, ContextFormat

# Тоже работает
from rtmdk.memory.core import RTMDKConfig, RTMDKMemory
from rtmdk.config import RTMDKConfig  # тот же класс (re-export)

# Неправильно — больше нет отдельного dataclass в config.py
# from rtmdk.config import RTMDKConfig as ConfigProfiles  # удалено
```

---

*Последнее обновление: Апрель 2026, RTMDK v8.1, unified config architecture*
