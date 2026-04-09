# RTMDK Documentation

> Resonance-Topological Memory for LLMs — Version 8.0

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

---

## Конфигурация — 8 профилей

| Профиль | RAM | Latency | Nodes | Назначение |
|---------|:---:|:---:|:---:|---|
| `RTMDKConfig.local()` | ~16MB | ~5ms | 10K | Персональный ассистент |
| `RTMDKConfig.production()` | ~50MB | ~6ms | 100K | Мультипользовательский сервер |
| `RTMDKConfig.research()` | ~200MB | ~50ms | ∞ | Максимальная точность |
| `RTMDKConfig.enterprise()` | ~250MB/shard | ~15ms | 500K+ | Распределённая система |
| `RTMDKConfig.agent()` | ~30MB | ~8ms | 50K | Автономный агент |
| `RTMDKConfig.legal()` | ~100MB | ~20ms | 200K | Юриспруденция (Z3) |
| `RTMDKConfig.medical()` | ~100MB | ~20ms | 200K | Медицина (Z3 + trust) |
| `RTMDKConfig.streaming()` | ~30MB | ~3ms | 50K | High-throughput real-time |

```python
from rtmdk.config import RTMDKConfig

config = RTMDKConfig.production()  # или local(), research(), etc.
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
├── config.py          # Центральная конфигурация (8 профилей)
├── nodes.py           # Data-классы узлов
├── engrams.py         # Phase 18: Энграммы
├── utils/             # Утилиты (modality, attention, formatting)
├── engines/           # Движки
│   ├── causal_traversal.py    # Причинный обход
│   ├── ssm_dynamics.py        # State Space Models (Mamba)
│   ├── trust_consensus.py     # DAG доверия
│   └── neuro_symbolic_prover.py # Z3/Prolog
├── support/           # 24 класса поддержки
└── production/        # Production модули
    ├── offline_dreamer.py      # Фоновые циклы
    ├── query_cache.py          # LRU кэш запросов
    ├── bm25_fallback.py        # BM25 fallback
    ├── active_inference.py     # Curiosity loop (research)
    ├── adversarial_arena.py    # Self-play тесты (research)
    └── tpr.py                  # Tensor Product Rep (research)
```

---

*Последнее обновление: Апрель 2026, RTMDK v8.0, 25 коммитов, 75+ файлов, 25,000+ строк кода*
