# Архитектура RTMDK v8.0

> Полный обзор всех фаз, модулей и компонентов системы

---

## Обзор системы

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RTMDK v8.0 Architecture                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  API Layer      │  │  Config Layer   │  │  Production Layer   │  │
│  │  rtmdk_server   │  │  8 profiles     │  │  Dreamer, Cache     │  │
│  │  OpenAI compat  │  │  local/prod/..  │  │  Trust, Prover      │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────┬───────────┘  │
│           │                    │                      │              │
│  ┌────────┴────────────────────┴──────────────────────┴───────────┐  │
│  │                     RTMDKMemory (Monolith)                      │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐  │  │
│  │  │ RTMDKField│ │ EngramMgr │ │ Causal    │ │ SSM Dynamics  │  │  │
│  │  │ (nodes)   │ │ (Phase 18)│ │ Traversal │ │ (Phase 19)    │  │  │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────────┘  │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────┐  │  │
│  │  │ HNSW      │ │ BM25      │ │ Symbolic  │ │ Trust         │  │  │
│  │  │ Index     │ │ Fallback  │ │ Overlay   │ │ Consensus     │  │  │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     External Integrations                       │  │
│  │  LM Studio  │  OpenRouter  │  OpenAI  │  Anthropic  │  Groq   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     Docker Deployment                           │  │
│  │  CPU (~200MB image)  │  GPU (~4GB, CUDA 12.1)                  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Фазы реализации

### Phase 1-14: Core RTMDK

| Компонент | Описание | Файл |
|-----------|----------|------|
| **MemoryNode** | Узел памяти: фаза, амплитуда, салентность, латентная позиция | `nodes.py` |
| **RTMDKField** | Поле памяти: резонанс, консолидация, decay | `rtmdk_memory_v8.py` |
| **Resonance** | K_spatial × K_phase × A × S — мера релевантности | `rtmdk_memory_v8.py` |
| **Consolidation** | Диалектическое слияние узлов с высоким напряжением | `rtmdk_memory_v8.py` |
| **HNSW** | Приближённый поиск O(log N) | `support/hnsw.py` |
| **BM25** | Текстовый поиск fallback | `support/bm25.py` |
| **IncPCA** | Инкрементальная проекция | `support/projection.py` |

### Phase 15: Version Control & Attention

| Компонент | Описание |
|-----------|----------|
| **VersionControl** | Дельта-версионирование состояний поля |
| **ProactiveClarification** | Генерация уточняющих вопросов при слабом резонансе |
| **AttentionTokens** | Форматирование контекста с attention score |

### Phase 16: Symbolic & Safety

| Компонент | Описание |
|-----------|----------|
| **SymbolicOverlay** | Извлечение логических правил из поля |
| **SafetyCertifier** | Lyapunov-стабильность поля |
| **UMP** | Universal Memory Protocol — стандартизация |

### Phase 17: Role Sharding

| Компонент | Описание |
|-----------|----------|
| **RoleShardRouter** | Маршрутизация по ролям узлов |
| **SwarmConsensus** | Консенсус между агентами |

### Phase 18: Энграммы

**Концепция:** Группа коактивированных узлов = одно воспоминание

```
EngramPattern                    EngramIndex                    PatternCompleter
┌──────────────────┐            ┌──────────────────┐            ┌──────────────────┐
│ id: str          │            │ search(query)    │            │ complete(query,  │
│ node_weights: {} │◄────────── │ update_centroid()│            │   engrams)       │
│ centroid: np.arr │            │ size             │            │ min_overlap=0.2  │
│ strength: float  │            └──────────────────┘            └──────────────────┘
│ activation_count │
│ semantic_core    │            EngramManager
│ context_tags: {} │            ┌──────────────────┐
│ tier: str        │───────────►│ create_engram()  │
└──────────────────┘            │ retrieve()       │
                                │ decay()          │
                                │ merge_overlaps() │
                                │ expand_engrams() │
                                └──────────────────┘
```

**Результат:** Recall@1 94% → 95.2%, Search speed в 3x быстрее

**Файл:** `rtmdk/engrams.py`

### Phase 19: Advanced Improvements

#### Фаза 1: Критические

| Модуль | Описание | Влияние |
|--------|----------|---------|
| **OfflineDreamer** | Фоновые циклы для TDA, кристаллизации, topology repair | -90% latency spikes |
| **CausalTraversal** | BFS по каузальному графу от top-K resonance узлов | +15-25% на "почему"-вопросах |
| **SSMDynamics** | State Space Models (Mamba): O(N) вместо O(N³) | Масштабирование до 1M+ узлов |

#### Фаза 2: Важные

| Модуль | Описание | Влияние |
|--------|----------|---------|
| **TrustConsensus** | DAG доверия + репутационные веса | Byzantine fault tolerance |
| **NeuroSymbolicProver** | Z3/Prolog для разрешения противоречий | Логическая консистентность |

#### Фаза 3: Исследовательские

| Модуль | Описание | Статус |
|--------|----------|:---:|
| **ActiveInference** | Curiosity-driven exploration | ✅ Research mode |
| **TPR** | Tensor Product Representations | ✅ Research mode |
| **AdversarialArena** | Self-play robustness testing | ✅ Research mode |

---

## 8 Профилей конфигурации

```python
from rtmdk.config import RTMDKConfig

# Персональный ассистент
config = RTMDKConfig.local()
# RAM: ~16MB, Latency: ~5ms, Nodes: до 10K

# Продакшен сервер
config = RTMDKConfig.production()
# RAM: ~50MB, Latency: ~6ms, Nodes: до 100K

# Исследования
config = RTMDKConfig.research()
# RAM: ~200MB, Latency: ~50ms, Nodes: unlimited

# Enterprise (100K+)
config = RTMDKConfig.enterprise()
# RAM: ~250MB/shard, Latency: ~15ms, Nodes: 500K+

# Автономный агент
config = RTMDKConfig.agent()
# С active inference, causal traversal

# Юриспруденция
config = RTMDKConfig.legal()
# Z3 prover, causal max_hops=5

# Медицина
config = RTMDKConfig.medical()
# Z3 prover, trust_min_reputation=0.5

# Streaming (минимальная задержка)
config = RTMDKConfig.streaming()
# RAM: ~30MB, Latency: ~3ms
```

---

## Поток данных

```
User Query
    │
    ▼
┌─────────────┐
│  Embedder   │  (LM Studio / OpenRouter / OpenAI)
│  768D vector│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│              EngramIndex (fast)                  │
│  O(log E) поиск по центроидам энграмм           │
│  Если найдено → expand to nodes                 │
│  Если нет → fallback to field.query()           │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          CausalTraversalEngine                    │
│  BFS по каузальному графу от top-K узлов        │
│  Расширяет результаты причинными связями         │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           Context Formatter                       │
│  ATTENTION / COGNITIVE / PLAIN format            │
│  Optimization: 50-300 tokens adaptive            │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│              LLM Response                        │
│  Системный промпт + RTMDK контекст → ответ      │
└─────────────────────────────────────────────────┘
```

---

## Docker Deployment

### CPU Image (~200MB)
```dockerfile
FROM python:3.10-slim
# curl, requirements.txt, application code
EXPOSE 8080
```

### GPU Image (~4GB)
```dockerfile
FROM python:3.10-slim
# + PyTorch CUDA 12.1
EXPOSE 8080
```

### docker-compose.yml
```yaml
services:
  rtmdk-api:
    build: .
    ports: ["8080:8080"]
    env_file: .env
    # 19 environment variables for all providers
    volumes: [rtmdk-data:/data]
```

### Поддерживаемые API провайдеры
| Провайдер | Переменная |
|-----------|-----------|
| LM Studio | `RTMDK_API_PROVIDER=lm_studio` |
| OpenRouter | `RTMDK_API_PROVIDER=openrouter` |
| OpenAI | `RTMDK_API_PROVIDER=openai` |
| Anthropic | `RTMDK_API_PROVIDER=anthropic` |
| Custom (Groq, Together, LocalAI) | `RTMDK_API_PROVIDER=custom` |

---

## Ресурсные затраты

### RAM по масштабу

| N узлов | Без энграмм | С энграммами | Delta |
|---------|:---:|:---:|---|
| 1,000 | 14 MB | 16 MB | +2 MB |
| 10,000 | 80 MB | 90 MB | +10 MB |
| 100,000 | 750 MB | 780 MB | +30 MB |
| 1,000,000 | 7.5 GB | 7.6 GB | +100 MB |

### CPU по операциям

| Операция | Время | Зависимость |
|----------|:---:|---|
| Embedding (LM Studio) | ~80ms | Сеть |
| Engram search | ~5ms | O(log E) |
| Causal traversal | ~3ms | O(K·d^hops) |
| Context formatting | ~1ms | O(top_k) |
| **Итого P50** | **~89ms** | |

---

## Статистика проекта

| Метрика | Значение |
|---------|----------|
| **Коммитов** | 24 |
| **Файлов** | 75+ |
| **Строк кода** | 25,000+ |
| **Модулей** | 28 |
| **Публичных API** | 105+ |
| **Профилей** | 8 |
| **Phases** | 19 |
| **Документации** | 8 файлов |

---

*Документ создан: Апрель 2026, RTMDK v8.0*
