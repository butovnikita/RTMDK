# Архитектура RTMDK v8.3

> Полный обзор всех фаз, модулей и компонентов системы
> Обновлено после Leadership Cleanup (v8.3-alpha) — декомпозиция monolithic field.py / core.py

---

## Обзор системы

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RTMDK v8.3 Architecture                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  API Layer      │  │  Config System  │  │  Production Layer   │  │
│  │  rtmdk_server   │  │  Unified Config │  │  Dreamer, Cache     │  │
│  │  OpenAI compat  │  │  8 presets      │  │  Trust, Prover      │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────┬───────────┘  │
│           │                    │                      │              │
│           │             ┌──────┴──────┐               │              │
│           │             │ Env Vars    │               │              │
│           │             │ 59 overrides│               │              │
│           │             │ /api/config │               │              │
│           │             └──────┬──────┘               │              │
│  ┌────────┴────────────────────┴──────────────────────┴───────────┐  │
│  │                     RTMDKMemory (Facade)                        │  │
│  │         Delegates to: ContextManager, PipelineBuilder           │  │
│  │         Backlog: BacklogModulesInitializer, MemoryPostInit      │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┴───────────────────────────────────┐  │
│  │                     RTMDKField (Coordinator)                    │  │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐  │  │
│  │   │NodeMgr  │ │QueryMgr │ │TopoMgr  │ │AsyncMgr │ │Sched   │  │  │
│  │   │CrystMgr │ │IndexMgr │ │ProjMgr  │ │OperMgr  │ │CognMgr │  │  │
│  │   │MergeMgr │ │RoutMgr  │ │ConsolMgr│ │EngramMgr│ │LearnMgr│  │  │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘  │  │
│  │   FieldInitializer wires all subsystems at construction time   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     External Integrations                       │  │
│  │  LM Studio  │  OpenRouter  │  OpenAI  │  Anthropic  │  Groq   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                     Docker Deployment                           │  │
│  │  Production (~200MB)  │  Home (~400MB)  │  GPU (~4GB, CUDA 12) │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Фазы реализации

### Phase 1-14: Core RTMDK

| Компонент | Описание | Файл |
|-----------|----------|------|
| **MemoryNode** | Узел памяти: фаза, амплитуда, салентность, латентная позиция | `memory/core.py` (inline), `nodes.py` (standalone) |
| **RTMDKField** | Поле памяти: маршрутизация, консолидация, decay | `memory/field.py` |
| **ResonanceEngine** | K_spatial × K_phase × A × S — чистая математика резонанса | `memory/resonance.py` |
| **NodeCacheManager** | Предвычисленные numpy-массивы для векторизованного query | `memory/cache_manager.py` |
| **IndexManager** | HNSW + BM25 + sparse shard routing | `memory/index_manager.py` |
| **ProjectionManager** | Projection + SOT tokenizer lifecycle | `memory/projection_manager.py` |
| **ConsolidationManager** | Диалектическое слияние узлов с высоким напряжением | `memory/consolidation_manager.py` |
| **QueryManager** | Все пути query/retrieval + batch resonance | `memory/query_manager.py` |
| **RoutingManager** | Shard routing и обновление shard centers | `memory/routing_manager.py` |
| **TopologyManager** | Tension, soft gates, pruning, integrity | `memory/topology_manager.py` |
| **AsyncPipelineManager** | Background async workers (evolve, save) | `memory/async_pipeline_manager.py` |
| **CrystallizationManager** | Recurring pattern detection (DBSCAN) | `memory/crystallization_manager.py` |
| **NodeManager** | Node ingestion, batch add, delete, queue | `memory/node_manager.py` |
| **CognitiveManager** | Self-supervision, TDA, state encoding, compression | `memory/cognitive_manager.py` |
| **Riemannian Geometry** | Операции на шаре Пуанкаре (exp/log/midpoint/scalar) | `memory/geometry.py`, `utils/hyperbolic.py` |
| **Spectral Clustering** | Спектральный графовый Laplacian для глобальной кластеризации перед merge | `memory/spectral.py` |
| **Kalman Filter** | EKF для отслеживания неопределённости позиций узлов | `memory/kalman.py` |
| **Conformal Prediction** | ICP-калибровка score-threshold для гарантий coverage | `memory/conformal.py` |
| **Local Bandwidth** | Адаптивная k-NN KDE ширина ядра per-node | `memory/field.py` (cache) |
| **HNSW** | Приближённый поиск O(log N) | `support/hnsw.py`, `memory/index_manager.py` |
| **BM25** | Текстовый поиск fallback | `support/bm25.py`, `memory/index_manager.py` |
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

### Phase 19: Advanced Improvements (Mathematical Enhancement Track — Completed)

#### P0 — Critical Fixes

| Модуль | Описание | Влияние | Статус |
|--------|----------|---------|:------:|
| **Riemannian SGD** | Riemannian gradients + exp_map на шаре Пуанкаре; устраняет boundary-clamp | <1% clamped nodes | ✅ |

#### P1 — Production Quality

| Модуль | Описание | Влияние | Статус |
|--------|----------|---------|:------:|
| **Conformal Prediction** | ICP threshold с гарантией coverage ≥ 1−α | −15…25% false-context | ✅ |
| **Local Adaptive Bandwidth** | k-NN KDE per-node σᵢ | +10…15% MRR | ✅ |

#### P2 — Stability & Observability

| Модуль | Описание | Влияние | Статус |
|--------|----------|---------|:------:|
| **Spectral Laplacian** | Глобальная кластеризация high-tension узлов перед merge | >0.70 purity | ✅ |
| **Kalman Filter** | EKF неопределённости; score damping по tr(Σ) | outlier-resistant | ✅ |
| **OfflineDreamer** | Фоновые циклы для TDA, кристаллизации, topology repair | −90% latency spikes | ✅ |
| **CausalTraversal** | BFS по каузальному графу от top-K resonance узлов | +15-25% на "почему"-вопросах | ✅ |
| **SSMDynamics** | State Space Models (Mamba): O(N) вместо O(N³) | Тестировано до 10K узлов; теоретическое масштабирование до 1M+ | ✅ |

#### Фаза 2: Важные

| Модуль | Описание | Влияние |
|--------|----------|---------|
| **TrustConsensus** | DAG доверия + репутационные веса | Byzantine fault tolerance *(research prototype, not production-tested)* |
| **NeuroSymbolicProver** | Z3/Prolog для разрешения противоречий | Логическая консистентность |

#### Фаза 3: Исследовательские

| Модуль | Описание | Статус |
|--------|----------|:---:|
| **ActiveInference** | Curiosity-driven exploration | ✅ Research mode |
| **TPR** | Tensor Product Representations | ✅ Research mode |
| **AdversarialArena** | Self-play robustness testing | ✅ Research mode |

### Phase 20: Domain Memory & Concept Lifecycle

| Компонент | Описание | Файл |
|-----------|----------|------|
| **Domain Classifier** | Pattern-based классификация доменов (~0.1ms) | `utils/domain_classifier.py` |
| **Domain Fields** | `domain`, `subdomain`, `topic` поля в узле | `nodes.py` |
| **Concept Lifecycle** | `state`, `confidence`, `revision_count`, `conflict_with` | `nodes.py` |
| **Evidence Spans** | Traceability для legal/medical | `nodes.py` |
| **Cross-domain Guard** | Запрет консолидации узлов из разных доменов | `core.py` |

### Phase 21: Self-Organizing Tokenizer + Embedding Field (SOT)

Заменяет статический `nn.Embedding` на динамическое поле, которое учится контрастным Хеббом, растёт от байт к субтокенам и синхронизируется с SSM-динамикой.

| Компонент | Описание | Файл |
|-----------|----------|------|
| **SOTokenizer** | Байт → субтокен или word-level токенизатор. Vocab растёт через co-retrieval merges (byte mode) или накопление слов (word mode). Поддерживает `token_dim != latent_dim` через learnable projection. | `memory/self_organizing_field.py` |
| **ContrastiveHebbian** | Online contrastive learning: positives pull closer, negatives push apart. Hard negatives: выбираются ближайшие non-positives. | `memory/self_organizing_field.py` |
| **EmbeddingFieldSSM** | SSM-моментум для плавных траекторий. Диагональный режим O(N·d) позволяет масштабировать `latent_dim` без просадок. | `memory/self_organizing_field.py` |
| **CooccurrenceStore** | Bounded co-occurrence dict: max 100K entries, auto-prune по lowest weight. | `memory/self_organizing_field.py` |
| **bootstrap_sbert** | Offline SBERT bootstrap: ridge regression от word counts → SBERT space. | `memory/bootstrap_sbert.py` |
| **bootstrap_fasttext** | FastText/GloVe bootstrap: копирование word vectors, PCA projection. 0.23s vs 12s SBERT. | `memory/bootstrap_fasttext.py` |
| **SOT Integration** | `step()` / `query()` / `add_node()` адаптированы для любой размерности эмбеддингов. Auto-bootstrap при init. | `memory/core.py` |

**Режимы токенизации:**
- **Byte mode** (default): 256 байт + merges. Компактный, но семантически слепой ("earthquake" и "quake" не связаны).
- **Word mode**: split по whitespace/punctuation. Vocab растёт до `max_vocab`. Требует bootstrap для cold-start.

**SBERT Bootstrap pipeline:**
1. `python -m rtmdk bootstrap corpus.json -o bootstrap.npz` — offline генерация projection
2. `RTMDKConfig(sot_bootstrap_projection="bootstrap.npz")` — загрузка при старте
3. Или `sot_bootstrap_corpus="corpus.json"` — auto-bootstrap при инициализации

**Ключевые свойства:**
- **Автономность**: нет зависимости от внешнего embedder API для query.
- **Адаптивность**: vocab растёт под домен поля (merge по co-retrieval, не по corpus frequency).
- **Пластичность**: эмбеддинги нод и токенов дрейфуют в ответ на usage через Hebbian updates.
- **Плавность**: SSM sync даёт инерцию обновлениям, предотвращая резкие скачки.
- **Масштабируемость**: `token_dim=256` + `latent_dim=64` даёт высокую ёмкость токенов при быстром поле. Диагональный SSM убирает O(d²) bottleneck.

**Флаги конфигурации:** `sot_enabled`, `sot_token_dim`, `sot_max_vocab`, `sot_contrastive_lr`, `sot_ssm_sync`, `sot_diagonal_ssm`, `sot_use_for_query`, `sot_merge_freq`, `sot_merge_threshold`, `sot_tokenization_mode`, `sot_warm_start_corpus`, `sot_subword_seed`, `sot_attention_pooling`, `sot_hard_negatives`, `sot_retrieval_feedback`, `sot_skipgram_window`, `sot_bootstrap_projection`, `sot_bootstrap_corpus`, `sot_bootstrap_model`, `sot_max_cooccurrence`.

---

### Phase 22: v8.2.1 Enhancement Wave

| Компонент | Описание | Файл |
|-----------|----------|------|
| **GraphQL API** | Strawberry GraphQL schema: Query (health, node, nodes) + Mutation (create_node, delete_node). Endpoint `/graphql`. | `server/graphql_schema.py` |
| **WebSocket Streaming** | Real-time WebSocket `/ws/memory` для query, ping/pong, live events. | `server/app.py` |
| **SOT Vocabulary Endpoint** | REST `/v1/sot/vocab` с пагинацией и поиском. | `server/app.py` |
| **SOT Persistence** | SOT tokenizer state сериализуется в `field_to_dict()` и загружается при `field_from_file()`. | `memory/serialization.py` |
| **SOT Graceful Degradation** | LRU eviction вместо RuntimeError при переполнении vocab. | `memory/self_organizing_field.py` |
| **SOT Circuit Breaker** | AsyncCircuitBreaker защищает SBERT bootstrap от cascading failures. | `server/app.py` |
| **React Admin Panel** | Vite + React приложение: Dashboard, Memory Nodes, Query, SOT. | `admin/` |
| **Vector-Native Storage** | Stub для SQLite-VSS / pgvector backend. | `production/vector_storage.py` |
| **Multi-Master Replication** | Stub для Raft/Paxos distributed consensus. | `production/replication.py` |

---
---

### Phase 23: Leadership Cleanup — Architecture Decoupling (v8.3.0)

**Цель:** Разбить монолитные `RTMDKField` (5265 строк) и `RTMDKMemory` (2603 строка) на сфокусированные подсистемы, устранить дублирование, упростить тестирование и onboarding.

**Результат:**
- `RTMDKField` → 844 строк (−84%) — координатор, делегирует всю работу менеджерам
- `RTMDKMemory` → ~1380 строк (−47%) — фасад + pipeline builder
- 21 извлечённый компонент

| Компонент | Откуда извлечён | Файл | Строки | Ответственность |
|-----------|----------------|------|--------|-----------------|
| **FieldInitializer** | `RTMDKField.__init__` (~460 строк) | `memory/field_initializer.py` | ~460 | Проводка всех подсистем: движки, индексы, менеджеры, security, scheduler |
| **ContextManager** | `RTMDKMemory.save_context`, `_retrieve_and_format` | `memory/context_manager.py` | ~160 | Сохранение контекста, retrieval pipeline, proactive clarification, symbolic overlay |
| **MemoryPostInitializer** | `RTMDKMemory.model_post_init` (~160 строк) | `memory/memory_post_initializer.py` | ~160 | Проводка backlog-модулей после создания поля: async workers, engrams, causal traversal |
| **BacklogModulesInitializer** | `RTMDKMemory._init_backlog_modules` (~95 строк) | `memory/backlog_modules_initializer.py` | ~95 | Инициализация production-модулей: Dreamer, Prover, Trust, Cache, MCTS |
| **PipelineBuilder** | `RTMDKMemory.build_pipeline`, `_attach_breaker` | `memory/pipeline_builder.py` | ~120 | Конструирование PipelineExecutor с circuit breakers и health checks |
| **OperationalManager** (расширен) | `RTMDKField._compress_field`, `_self_heal`, calibrate, rollback | `memory/operational_manager.py` | ~195 | Компрессия поля, rollback, intervention, causal summary, self-healing |

**Менеджеры (Phase 1–14, расширены в Cleanup):**
- `NodeManager`, `QueryManager`, `TopologyManager`, `AsyncPipelineManager`
- `CrystallizationManager`, `MergeManager`, `RoutingManager`
- `IndexManager`, `ProjectionManager`, `ConsolidationManager`
- `CognitiveManager`, `Scheduler`, `EngramManager`

**Ключевые архитектурные решения:**
1. **FieldInitializer как constructor-injection** — все зависимости `RTMDKField` создаются в одном месте, упрощая тестирование с моками.
2. **Thin wrappers + `__getattr__`** — публичный API `RTMDKMemory` сохранён полностью. Удалённые методы проксируются через `__getattr__` или делегируют в менеджеры.
3. **No-deletion policy** — Экспериментальный код сохранён; устаревшие флаги депрекированы, но оставлены для backward compatibility v8.x.
4. **Import cycle resolution** — `MemoryNode` импортируется через `rtmdk.nodes`, а не из `rtmdk.memory.core`, разрывая циклическую зависимость.

**Backward compatibility:**
- Все публичные методы `RTMDKField` и `RTMDKMemory` работают без изменений
- `ContextFormat` и `SecurityViolationError` re-export из `rtmdk.memory.core` для старых импортов

---
---

## Конфигурационная система (Unified Config)

RTMDK v8.3 использует **единый** `RTMDKConfig` dataclass из `rtmdk/memory/config.py`:

```
rtmdk/memory/config.py   ← ЕДИНСТВЕННЫЙ RTMDKConfig dataclass (~150 полей)
        │
        ├── re-export через rtmdk/config.py (пресеты)
        ├── re-export через rtmdk/__init__.py (main package)
        │
        └── __post_init__: 59 env var overrides (RTMDK_LATENT_DIM, etc.)
```

**Приоритет конфигурации:**
1. Явные аргументы в коде (`RTMDKConfig(latent_dim=128)`)
2. Env vars (`RTMDK_LATENT_DIM=128`)
3. Defaults пресета (`RTMDKConfig.local()`)
4. Defaults dataclass (`latent_dim=64`)

### 8 Пресетов

| Пресет | RAM | Latency | Nodes | Назначение |
|--------|:---:|:---:|:---:|---|
| `local()` | ~16MB | ~5ms | 10K | Персональный ассистент |
| `production()` | ~50MB | ~6ms | 100K | Мультипользовательский сервер |
| `research()` | ~200MB | ~50ms | ∞ | Максимальная точность |
| `enterprise()` | ~250MB/shard | ~15ms | 500K+ | Распределённая система |
| `agent()` | ~30MB | ~8ms | 50K | Автономный агент |
| `legal()` | ~100MB | ~20ms | 200K | Юриспруденция (Z3) |
| `medical()` | ~100MB | ~20ms | 200K | Медицина (Z3 + trust) |
| `streaming()` | ~30MB | ~3ms | 50K | High-throughput real-time |

### Runtime Configuration

```bash
# Через env vars
RTMDK_PRESET=production RTMDK_LATENT_DIM=128 python rtmdk_server.py

# Через API
curl -X POST http://localhost:8080/api/config \
  -d '{"RTMDK_PRESET": "production", "RTMDK_LATENT_DIM": "128"}'
# Ответ: {"status":"ok", "needs_restart": true, ...}
```

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

## Модульная структура (post-audit)

```
rtmdk/
├── __init__.py          # Публичный API
├── cli.py               # CLI интерфейс
├── config.py            # 8 пресетов RTMDKConfig
│
├── memory/              # Core kernel
│   ├── field.py              # RTMDKField (~840 lines, coordinator facade)
│   ├── core.py               # RTMDKMemory (~1380 lines, facade + pipeline)
│   ├── field_initializer.py  # FieldInitializer (~460 lines, subsystem wiring)
│   ├── context_manager.py    # ContextManager (~160 lines, save/load context)
│   ├── memory_post_initializer.py # MemoryPostInitializer (~160 lines, backlog wiring)
│   ├── backlog_modules_initializer.py # BacklogModulesInitializer (~95 lines, production modules)
│   ├── pipeline_builder.py   # PipelineBuilder (~120 lines, circuit breaker assembly)
│   ├── node_manager.py       # NodeManager (add/delete/batch/queue nodes)
│   ├── query_manager.py      # QueryManager (retrieval + batch resonance)
│   ├── topology_manager.py   # TopologyManager (tension, soft gates, pruning)
│   ├── async_pipeline_manager.py # AsyncPipelineManager (bg workers, evolve, save)
│   ├── crystallization_manager.py # CrystallizationManager (DBSCAN recurring patterns)
│   ├── merge_manager.py      # MergeManager (learned consolidation + quant pruning)
│   ├── routing_manager.py    # RoutingManager (shard routing + center updates)
│   ├── cognitive_manager.py  # CognitiveManager (self-supervision, TDA, compression)
│   ├── operational_manager.py # OperationalManager (calibrate, rollback, compress)
│   ├── resonance.py          # ResonanceEngine (math extraction)
│   ├── cache_manager.py      # NodeCacheManager (cache extraction)
│   ├── index_manager.py      # IndexManager (HNSW+BM25+shards extraction)
│   ├── projection_manager.py # ProjectionManager (projection + SOT lifecycle)
│   ├── consolidation_manager.py # ConsolidationManager (merge + spectral clustering)
│   ├── scheduler.py          # StepScheduler (periodic task orchestration)
│   ├── geometry.py           # Пуанкаре-операции (exp/log/midpoint/scalar)
│   ├── conformal.py          # ICP калибровка retrieval confidence
│   ├── spectral.py           # Spectral Graph Laplacian для consolidation
│   ├── kalman.py             # Riemannian EKF неопределённости узлов
│   ├── serialization.py      # Экспорт/импорт полей (msgpack/zlib/JSON)
│   ├── snapshot.py           # Дельта-версионирование
│   ├── wal.py                # Write-Ahead Log (fsync, crash recovery)
│   ├── distributed_lock.py   # File-based + Redis distributed locking
│   ├── observability.py      # MemoryMetrics + Prometheus export
│   ├── rag_quality.py        # Query decomposition + sentence reranking
│   ├── safety.py             # PoisonedMemoryDetector + rollback
│   ├── learned_consolidation.py # Differentiable consolidation
│   └── __init__.py
│
├── nodes.py             # Standalone dataclasses (MemoryNode, CausalEdge)
│
├── engines/             # Специализированные движки
│   ├── dreamer.py       # OfflineDreamer (фоновая оптимизация)
│   ├── causal.py        # CausalTraversalEngine
│   ├── consensus.py     # SwarmConsensusProtocol
│   ├── ssm.py           # SSMDynamics (Mamba)
│   ├── symbolic.py      # SymbolicOverlay + NeuroSymbolicProver
│   └── trust.py         # TrustConsensus
│
├── production/          # Production layer
│   ├── analytics_engine.py   # SQLite analytics
│   ├── cache_manager.py      # LRU кэш
│   ├── trust_scorer.py       # Репутационные веса
│   └── prover_factory.py     # Z3 / Prolog интеграция
│
├── support/             # Индексы и утилиты
│   ├── hnsw.py          # HNSW индекс O(log N)
│   ├── bm25.py          # Текстовый fallback
│   └── projection.py    # IncPCA
│
├── utils/               # Хелперы
│   └── domain_classifier.py  # Pattern-based domain detection
│
└── experimental/        # Исследовательские модули (опциональные)
    ├── tpr.py           # Tensor Product Representations
    ├── adversarial_arena.py  # Self-play robustness
    └── active_inference.py   # Curiosity-driven exploration
```

## Тестовое покрытие

| Модуль | Тесты | Статус |
|--------|:-----:|:------:|
| Security | `test_security.py` (9 тестов) | ✅ Pass |
| MemoryNode | `test_nodes.py` (5 тестов) | ✅ Pass |
| Phase 20 Domain | `test_domain_memory.py` (13 тестов) | ✅ Pass |
| Analytics | `test_rtmdk_eval.py` (9 тестов) | ✅ Pass |
| Swarm Consensus | `test_rtmdk_swarm.py` (10 тестов) | ✅ Pass |
| Riemannian SGD | `test_riemannian_consolidate.py` (12 тестов) | ✅ Pass |
| Conformal Prediction | `test_conformal_prediction.py` (10 тестов) | ✅ Pass |
| Local Bandwidth | `test_local_bandwidth.py` (10 тестов) | ✅ Pass |
| Spectral Laplacian | `test_spectral_consolidation.py` (13 тестов) | ✅ Pass |
| Kalman Filter | `test_kalman_filter.py` (15 тестов) | ✅ Pass |
| Chunked Query | `test_chunked_query.py` (1 тест) | ✅ Pass |
| SOT / Hebbian | `test_sot_*.py` (88 тестов) | ✅ Pass |
| Memory Leak | `test_memory_leak.py` (6 тестов) | ✅ Pass |
| Config Matrix | `test_config_matrix.py` (20 тестов) | ✅ Pass |
| Circuit Breaker | `test_circuit_breaker.py` (7 тестов) | ✅ Pass |
| Observability | `test_observability.py` (4 тестов) | ✅ Pass |
| Plugins | `test_plugins.py` (2 тестов) | ✅ Pass |
| Serialization | `test_msgpack_serialization.py` (3 тестов) | ✅ Pass |
| Graph Index | `test_naive_graph_index.py` (4 тестов) | ✅ Pass |
| Proxy | `test_proxy.py` (1 тест) | ✅ Pass |

**Итого: 261 тест, все проходят (pytest, ~43 сек).**

## Mathematical Enhancements Track (P0–P2)

С апреля 2026 реализован математический трек улучшений ядра системы:

### P0.1 Riemannian Geometry
- **Геометрия Пуанкаре** — пространство отрицательной кривизны для позиций узлов
- **Операции**: `mobius_add`, `mobius_scalar`, `mobius_distance`, `log_map_poincare`, `exp_map_poincare`
- **Применение**: `consolidate()` выполняет merge на касательном пространстве, `query()` учитывает расстояние в шаре Пуанкаре
- **Файл**: `rtmdk/geometry.py` | **Тесты**: `test_riemannian_consolidate.py`

### P1.2 Local Adaptive Bandwidth (k-NN KDE)
- **Идея**: per-node ширина ядра пропорциональна локальной плотности через k-NN расстояние
- **Формула**: σᵢ = σ_global × √(kdistᵢ / median(kdist))
- **Применение**: `_build_node_cache()` строит `_cached_bw` массив; `_compute_resonance_chunk()` использует per-node bw
- **Файл**: `memory/core.py` | **Тесты**: `test_local_bandwidth.py` (10 тестов)

### P1.1 Inductive Conformal Prediction (ICP)
- **Идея**: статистическая гарантия coverage для retrieval confidence
- **Алгоритм**: калибровка на non-conformity scores → quantile threshold → prediction set
- **Применение**: `query()` фильтрует результаты через `_apply_conformal_filter()`
- **Файл**: `memory/conformal.py` | **Тесты**: `test_conformal_prediction.py` (10 тестов)

### P2.1 Spectral Graph Laplacian
- **Идея**: глобальная кластеризация перед merge через спектральный анализ
- **Алгоритм**: affinity(W) → L_sym = I − D⁻¹/² W D⁻¹/² → eigengap → k-means на spectral embedding
- **Применение**: `consolidate()` вызывает `spectral_cluster_nodes()` перед greedy merge
- **Файл**: `memory/spectral.py` | **Тесты**: `test_spectral_consolidation.py` (13 тестов)

### P2.2 Riemannian Extended Kalman Filter (EKF)
- **Идея**: оценка неопределённости позиции каждого узла в латентном пространстве
- **Режимы**: диагональное приближение (O(d)) или полная ковариация (O(d²))
- **Применение**: `add_node()` инициализирует ковариацию; `consolidation` обновляет через merge; `query()` взвешивает score по 1/(1+tr(Σ))
- **Файл**: `memory/kalman.py` | **Тесты**: `test_kalman_filter.py` (15 тестов)

### Суммарные метрики трека

| Показатель | Значение |
|------------|----------|
| Новых файлов | 3 (`conformal.py`, `spectral.py`, `kalman.py`) |
| Новых тестов | 48 |
| Строк кода (ядро) | ~1,200 |
| Время выполнения всех тестов | ~35 сек |
| Покрытие фичей | P0.1, P1.1, P1.2, P2.1, P2.2 |
| Статус | ✅ Завершён, интегрирован в RTMDKField |

## Статистика проекта

| Метрика | Значение |
|---------|----------|
| **Коммитов** | 24 |
| **Файлов** | 75+ |
| **Строк кода** | 25,000+ |
| **Модулей** | 28 |
| **Публичных API** | 105+ |
| **Профилей** | 8 |
| **Phases** | 21 (Mathematical Track P0–P2) |
| **Документации** | 8 файлов |
| **Тестов** | 1112 |

---

*Документ создан: Апрель 2026, RTMDK v8.1*
*Обновлён: Май 2026 (v8.3.0 release)*


## Backlog Modules (v8.3.0)

### EngramEmbeddingCache
- File: 
tmdk/memory/engram_cache.py`n- Purpose: Hot/warm/cold tiered cache for node embeddings to avoid TieredNodeStore disk scans.
- Config: sot.engram_cache_enabled, sot.engram_cache_max_hot, sot.engram_cache_max_warm`n
### Observability
- File: 
tmdk/memory/observability.py`n- Purpose: Latency histograms (p50/p95/p99), cache hit ratio, threshold alerting.
- Config: sot.observability_enabled`n
### DistributedLock
- File: 
tmdk/memory/distributed_lock.py`n- Purpose: File-based inter-process lock with intra-process thread safety.
- Config: sot.distributed_lock_path`n
### RAG Quality
- File: 
tmdk/memory/rag_quality.py`n- Purpose: Query decomposition, sentence-level reranking, explicit feedback loop.
- Config: sot.sentence_reranker_enabled, sot.query_decomposition_enabled, sot.feedback_loop_enabled`n


## Production Hardening (v8.3.0)

### Observability
- MemoryMetrics with latency percentiles, cache hit ratio, Prometheus export
- Alert handlers: Webhook, Slack, PagerDuty
- JSON structured logging via rtmdk.utils.json_logger

### Distributed Lock
- File-based (msvcrt/fcntl) and Redis backends
- Intra-process thread safety

### RAG Quality
- QueryDecomposer: heuristic AND-split + optional LLM-based decomposition
- SentenceReranker: sentence-level cosine with batch embedding
- FeedbackLoop: SOT embedder updates with JSON persistence
- QueryRewriter: auto-rewrite on low retrieval quality
- QueryIntentClassifier: factual/exploratory/conversational/comparative
- ResultExplainer: human-readable retrieval reasons

### Safety
- RollbackManager: snapshot and rollback memory state
- PoisonedMemoryDetector: anomaly detection for injected/spam nodes
- CircuitBreaker for embedder with automatic recovery

### Performance
- AsyncEmbedder: async batching wrapper for high throughput
- EngramEmbeddingCache: hot/warm/cold tiers with NPZ save/load
- Sparse PMI for SIF: scipy.sparse + TruncatedSVD for vocab > 5000

### UX
- MemoryTimeline: chronological session view
- MemoryNarrator: story generation and Markdown export

