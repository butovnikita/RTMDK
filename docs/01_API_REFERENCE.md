# RTMDK — Полная документация

> Версия 8.1 | Модульная архитектура | 45+ модулей | 27 UX-функций | 100+ файлов | Mathematical Enhancement Track (P0–P2) завершён

---

## Все API Endpoints (36+)

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Базовый health check |
| GET | `/v1/models` | Список моделей |
| POST | `/v1/chat/completions` | Чат с памятью (OpenAI-compatible) |
| POST | `/v1/embeddings` | Эмбеддинги |
| POST | `/v1/feedback` | Feedback Loop |
| POST | `/graphql` | GraphQL API (Strawberry) |
| WS | `/ws/memory` | WebSocket streaming для real-time query |
| POST | `/v1/session/save` | Сохранить сессию |
| POST | `/v1/session/load` | Загрузить сессию |
| GET | `/v1/session/list` | Список сессий |
| POST | `/v1/backup/create` | Создать бэкап |
| POST | `/v1/backup/restore` | Восстановить бэкап |
| GET | `/v1/backup/list` | Список бэкапов |
| POST | `/v1/import/json` | Импорт JSON |
| POST | `/v1/import/url` | Импорт URL |
| GET | `/v1/analytics` | Аналитика |
| GET | `/v1/health` | Детальный health check |
| GET | `/v1/metrics` | Prometheus metrics |
| GET | `/v1/export?format=md` | Экспорт (md/txt/json) |
| GET/POST/DELETE | `/v1/tags/{node_id}` | Теги |
| GET | `/v1/rate-limit` | Rate limit status |
| GET | `/v1/events` | SSE events |
| GET | `/v1/cache/stats` | Cache stats |
| POST | `/v1/cache/clear` | Clear cache |
| POST | `/v1/memory/save` | Сохранить контекст |
| POST | `/v1/memory/query` | Запросить память |
| POST | `/v1/memory/query_pipeline` | Запросить память (pipeline API с метриками и cost) |
| GET  | `/v1/memory/pipeline/stream` | SSE streaming pipeline stage events |
| GET  | `/v1/memory/pipeline/health` | Pipeline per-stage health status |
| GET  | `/v1/memory/pipeline/metrics` | Aggregated pipeline metrics |
| GET  | `/v1/memory/pipeline/plan` | Preview execution plan без выполнения |
| GET  | `/v1/memory/pipeline/dag` | Pipeline stage dependency graph |
| GET  | `/v1/memory/pipeline/prometheus` | Prometheus exposition format |
| GET  | `/v1/analytics/pipeline` | Pipeline analytics dashboard |
| POST | `/v1/memory/batch_query` | Batch query памяти |
| POST | `/v1/memory/nodes` | Создать ноду |
| GET | `/v1/memory/nodes/{id}` | Получить ноду |
| PUT | `/v1/memory/nodes/{id}` | Обновить ноду |
| DELETE | `/v1/memory/nodes/{id}` | Удалить ноду |
| GET | `/v1/memory/nodes` | Список нод (пагинация) |
| POST | `/v1/memory/batch_ingest` | Batch ingest документов |
| GET | `/v1/memory/export` | Экспорт памяти |
| POST | `/v1/memory/import` | Импорт памяти |
| POST | `/v1/memory/clear` | Очистить память |
| GET | `/v1/memory/stats` | Статистика |
| GET | `/v1/analytics/overview` | Dashboard overview |
| GET | `/v1/analytics/memory` | Memory analytics |
| GET | `/v1/analytics/events` | Event log |
| GET | `/v1/analytics/report` | Full report |
| POST | `/v1/analytics/track` | Track custom event |
| POST | `/v1/admin/api-keys` | Создать API ключ |
| GET | `/v1/admin/api-keys` | Список API ключей |
| POST | `/v1/admin/api-keys/revoke` | Отозвать ключ |
| DELETE | `/v1/admin/api-keys/{hash}` | Удалить ключ |
| GET | `/v1/admin/tenants` | Список тенантов |
| GET | `/v1/admin/audit-log` | Журнал аудита (admin) |
| GET | `/v1/admin/retention` | Статистика retention (admin) |
| POST | `/v1/webhooks` | Подписаться на webhook |
| DELETE | `/v1/webhooks/{id}` | Отписаться |
| GET | `/v1/webhooks` | Список подписок |

---

## Оглавление

1. [Обзор архитектуры](#1-обзор-архитектуры)
2. [Структура пакета](#2-структура-пакета)
3. [Установка и быстрый старт](#3-установка-и-быстрый-старт)
4. [Полный справочник конфигурации](#4-полный-справочник-конфигурации)
5. [Ядро: RTMDKMemory и RTMDKField](#5-ядро-rtmdkmemory-и-rtmdkfield)
6. [Утилиты](#6-утилиты)
7. [Движки (Engines)](#7-движки-engines)
8. [Поддержка (Support)](#8-поддержка-support)
9. [Агенты и Продакшен](#9-агенты-и-продакшен)
10. [EmbedderFactory](#10-embedderfactory)
11. [Сервер и HTTP API](#11-сервер-и-http-api)
12. [CLI-чат с LM Studio](#12-cli-чат-с-lm-studio)
13. [Тестирование](#13-тестирование)
14. [Обратная совместимость](#14-обратная-совместимость)
15. [Развёртывание](#15-развёртывание)
16. [История коммитов](#16-история-коммитов)

---

## 1. Обзор архитектуры

RTMDK (Resonance-Topological Memory) — система памяти для LLM, моделирующая память как **динамическое когнитивное поле**. Узлы памяти — не пассивные записи, а активные осцилляторы с фазой, амплитудой, салентностью и каузальными связями.

### Ключевые принципы

| Принцип | Описание |
|---------|----------|
| **Резонансный поиск** | Запрос возбуждает колебание в поле; ответ — интерференционная картина |
| **Диалектическая консолидация** | Объединение узлов через тезис+антитезис→синтез |
| **Каузальная топология** | Do-calculus, контрфактуалы, обнаружение противоречий |
| **Непрерывная динамика** | Neural ODE/SDE: dX/dt = F(X, u) + σ·dW |
| **Самовосстановление** | Обнаружение мёртвых зон, гиперконвергенции, фрагментации |
| **Мета-адаптивность** | Автоподстройка гиперпараметров по куртозису откликов |
| **Символический слой** | Probabilistic Horn clauses из консолидированных узлов |
| **Ролевое шардирование** | Изоляция контекстов с Kuramoto внутри шардов |

### Два стиля импорта

```python
# Модульный стиль (рекомендуется для новых проектов)
from rtmdk import RTMDKMemory, RTMDKConfig, MemoryNode

# Ядро напрямую
from rtmdk.memory.core import RTMDKMemory, RTMDKConfig, MemoryNode
```

---

## 2. Структура пакета

```
rtmdk/                              # Python-пакет (72+ публичных символа)
├── __init__.py                     # Re-export всех символов
├── config.py                       # RTMDKConfig + 5 enums (~340 строк)
├── nodes.py                        # 10 data-классов (~234 строки)
│
├── utils/                          # Утилиты
│   ├── __init__.py
│   ├── modality.py                 # detect_modality, detect_tier
│   ├── hyperbolic.py               # poincare_dist, exp/log_map, mobius_add
│   ├── attention.py                # apply_attention_bias, format_cognitive_context
│   └── formatting.py               # format_context, build_system_prompt
│
├── engines/                        # Движки (вычисления)
│   ├── __init__.py
│   ├── causal.py                   # CausalInferenceEngine
│   ├── predictive.py               # PredictiveCodingModel
│   ├── counterfactual.py           # ScenarioPlanner
│   ├── privacy.py                  # DifferentialPrivacy
│   └── neural_ode.py               # NeuralODEDynamics
│
└── support/                        # Поддержка (24+ класса)
    ├── __init__.py
    ├── meta_controller.py          # MetaController (Optuna/grid search)
    ├── kuramoto.py                 # KuramotoSync, FederatedRTMDK
    ├── meta_adaptive.py            # MetaAdaptiveKernel
    ├── healer.py                   # TopologyHealer
    ├── projection.py               # IncPCAProjection
    ├── bm25.py                     # BM25Index
    ├── threshold.py                # AdaptiveThreshold
    ├── tda.py                      # TDAMonitor
    ├── hnsw.py                     # HNSWIndex
    ├── torch_backend.py            # TorchBackend
    ├── learnable.py                # LearnableKernel, DifferentiableConsolidation
    ├── goal_tracker.py             # GoalTracker
    ├── rl_feedback.py              # RLFeedbackLoop
    ├── event_driven.py             # LowRankCompressor, EventDrivenScheduler
    ├── meta_memory.py              # MetaMemoryEvaluator
    ├── security.py                 # SecurityValidator
    ├── swarm.py                    # SwarmConsensusProtocol
    ├── agents/__init__.py          # AgentPlanner, HypothesisVerifier, ToolRouter
    ├── production/__init__.py      # ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager
    ├── version_control.py          # VersionControl, NodeDelta, Version, DiffResult (Phase 15)
    ├── entropy_controller.py       # EntropyController (Phase 15)
    ├── triton_backend.py           # TritonBackend (Phase 15)
    ├── symbolic_overlay.py         # SymbolicOverlay, SymbolicRule, ConflictDetector (Phase 16)
    ├── safety_certifier.py         # SafetyCertifier, LyapunovFunction (Phase 16)
    ├── ump.py                      # UniversalMemoryProtocol (Phase 16)
    └── role_shard_router.py        # RoleShardRouter, RoleShard, RoleDetector (Phase 17)
```

**Внешние файлы** (не входят в пакет `rtmdk/`):

| Файл | Описание |
|------|----------|
| `rtmdk/memory/core.py` | Ядро системы: RTMDKField + RTMDKMemory (~7000 строк) |
| `rtmdk_server.py` | OpenAI-compatible HTTP-сервер |
| `lmstudio_rtmdk_chat.py` | CLI-чат через LM Studio |
| `streamlit_app.py` | Интерактивный дашборд |
| `eval_pipeline.py` | Benchmarks: ContinualQA, LongBench, MemoryBench |
| `swarm_memory.py` | Мультиагентная роевая память |
| `tests/smoke_test.py` | Быстрая проверка критических путей |
| `embedder_factory.py` | Unified embedding интерфейс |

---

## 3. Установка и быстрый старт

### Зависимости

```bash
# Минимальные (ядро)
pip install numpy scipy pydantic

# Сервер
pip install fastapi uvicorn requests

# Разработка (тесты, дашборд, оптимизация)
pip install -r requirements-dev.txt
```

### Минимальный пример

```python
from rtmdk import RTMDKConfig, RTMDKMemory
import numpy as np

def embedder(text: str) -> np.ndarray:
    rng = np.random.default_rng(hash(text) % 2**32)
    return np.random.randn(768).astype(np.float32) * 0.1

config = RTMDKConfig(
    embedding_dim=768, latent_dim=64, top_k=3,
    causal_topological=True, self_healing=True, goal_tracking=True,
    attention_bias=True, version_control=True, symbolic_overlay=True,
)
memory = RTMDKMemory(config=config, embedder=embedder)
memory.save_context(
    {"input": "Меня зовут Никита, я разработчик", "session_id": "u1"},
    {"output": "Запомнил: Никита — разработчик"}
)
ctx = memory.load_memory_variables({"input": "Кто я?", "session_id": "u1"})
print(ctx["rtmdk_context"])
# → [ATTN:0.42|SAL:0.60|TIER:S] Меня зовут Никита, я разработчик
```

---

## 4. Полный справочник конфигурации

### Enums

| Enum | Значения | Назначение |
|------|----------|------------|
| `ConsolidationMode` | `DIALECTICAL`, `MERGE`, `PRUNE` | Стратегия объединения узлов |
| `Backend` | `NUMPY`, `TORCH` | Бэкенд вычислений |
| `ContextFormat` | `PLAIN`, `JSON`, `YAML`, `ATTENTION` | Формат контекста для LLM |
| `FieldHealth` | `STABLE`, `DEGRADED`, `CRITICAL`, `HEALING` | Состояние поля |
| `EvalMode` | `PRODUCTION`, `SHADOW`, `EVALUATION` | Режим оценки |

### RTMDKConfig — все параметры по фазам

#### Фаза 0: Базовые

| Параметр | Default | Описание |
|----------|---------|----------|
| `embedding_dim` | 768 | Размерность эмбеддинга |
| `latent_dim` | 64 | Размерность скрытого многообразия |
| `resonance_kernel` | "gaussian_phase" | Ядро резонанса: gaussian, cosine, gaussian_phase |
| `phase_coupling` | 0.3 | Сила фазового выравнивания |
| `bandwidth` | 1.0 | Ширина ядра резонанса |
| `attraction_lr` | 0.02 | Скорость притяжения к цели |
| `phase_sync_lr` | 0.01 | Скорость синхронизации фаз |
| `decay_rate` | 0.998 | Скорость затухания салентности |
| `min_amplitude` | 0.05 | Минимальная амплитуда узла |
| `tension_threshold` | 0.25 | Порог консолидации |
| `consolidation_mode` | `DIALECTICAL` | Режим консолидации |
| `max_nodes` | 5000 | Максимум узлов (None = без лимита) |
| `top_k` | 5 | Количество узлов в ответе |
| `min_response` | 0.1 | Минимальный резонанс для выдачи |
| `enable_async` | True | Включить async pipeline |
| `log_level` | "INFO" | Уровень логирования |

#### Фаза 1–2: Контекст и адаптивность

| Параметр | Default | Описание |
|----------|---------|----------|
| `context_format` | `ContextFormat.PLAIN` | Формат контекста |
| `use_structured_prompt` | True | Использовать структурированные промпты |
| `adaptive_threshold` | False | Скользящий порог консолидации |
| `adaptive_window` | 30 | Размер окна для адаптивного порога |
| `learn_projection` | False | IncPCA-обучение проекции |
| `projection_lr` | 0.001 | Скорость обучения проекции |
| `projection_update_freq` | 50 | Частота обновления проекции |
| `pca_n_components` | None (= latent_dim) | Количество компонент PCA |
| `bm25_fallback` | False | Текстовый поиск при промахе |
| `bm25_k1` | 1.5 | BM25 k1 параметр |
| `bm25_b` | 0.75 | BM25 b параметр |
| `soft_gates` | False | Sigmoid-ворота для узлов |
| `gate_temperature` | 0.15 | Температура sigmoid-ворот |
| `self_supervision` | False | Самообучение |
| `self_sup_threshold` | 0.3 | Порог самообучения |
| `self_sup_verify_after_consolidate` | False | Проверка после консолидации |
| `backend` | `Backend.NUMPY` | NUMPY или TORCH |
| `gpu_batch_size` | 512 | Размер батча для GPU |
| `l2_regularization` | 0.0001 | L2 регуляризация проекции |
| `false_merge_threshold` | 0.4 | Порог ложного слияния |
| `field_stability_window` | 20 | Окно стабильности поля |
| `enable_rollback` | False | Включить откат |
| `max_rollback_history` | 50 | Макс. история откатов |

#### Фаза 3: Мультимодальность + HNSW + TDA

| Параметр | Default | Описание |
|----------|---------|----------|
| `multimodal` | False | Мультимодальная обработка |
| `modalities` | ["text"] | Список модальностей |
| `modality_phase_shifts` | {} | Сдвиги фаз по модальностям |
| `use_hnsw` | False | Approximate nearest neighbor |
| `hnsw_m` | 16 | HNSW M параметр |
| `hnsw_ef_construction` | 200 | HNSW ef_construction |
| `tda_monitoring` | False | Топологический мониторинг |
| `tda_check_freq` | 50 | Частота TDA проверок |

#### Фаза 1 (Track): Differentiable field

| Параметр | Default | Описание |
|----------|---------|----------|
| `differentiable` | False | Дифференцируемое поле |
| `learnable_bandwidth` | False | Обучаемая ширина ядра |
| `learnable_phase_coupling` | False | Обучаемое фазовое сопряжение |
| `learnable_decay` | False | Обучаемый decay rate |
| `gradient_clip` | 1.0 | Клиппинг градиентов |
| `consolidation_loss_weight` | 0.1 | Вес потерь консолидации |

#### Фаза 5: Мета-адаптивность + Самовосстановление

| Параметр | Default | Описание |
|----------|---------|----------|
| `meta_adaptive` | False | MetaAdaptiveKernel |
| `meta_adaptation_lr` | 0.005 | Скорость мета-адаптации |
| `kurtosis_target_min` | 1.5 | Мин. целевой куртозис |
| `kurtosis_target_max` | 4.0 | Макс. целевой куртозис |
| `self_healing` | False | TopologyHealer |
| `healing_check_freq` | 25 | Частота проверок лечения |
| `dead_zone_threshold` | 0.15 | Порог мёртвой зоны |
| `hyperconvergence_threshold` | 0.05 | Порог гиперконвергенции |
| `fragmentation_threshold` | 0.6 | Порог фрагментации |
| `healing_strength` | 0.1 | Сила лечения |
| `max_healing_nodes_per_step` | 5 | Макс. узлов лечения за шаг |

#### Фаза 6: Каузально-топологическая память

| Параметр | Default | Описание |
|----------|---------|----------|
| `causal_topological` | False | Включить CausalInferenceEngine |
| `causal_discovery_min_samples` | 20 | Мин. сэмплов для discovery |
| `causal_p_threshold` | 0.05 | Порог p-value для PC-algorithm |
| `do_calculus_validation` | True | Валидация через do-calculus |
| `counterfactual_enabled` | False | Контрфактуальные запросы |
| `counterfactual_max_depth` | 3 | Макс. глубина рассуждений |
| `contradiction_detection` | True | Обнаружение противоречий |
| `contradiction_threshold` | 0.3 | Порог противоречий |
| `causal_adjustment_sets` | True | Backdoor adjustment sets |

#### Фаза 7: Neural ODE/SDE

| Параметр | Default | Описание |
|----------|---------|----------|
| `continuous_dynamics` | False | Непрерывная эволюция через ODE |
| `ode_solver` | "RK45" | Метод решения (RK45 / odeint) |
| `ode_atol` | 1e-6 | Абсолютная точность ODE |
| `ode_rtol` | 1e-5 | Относительная точность ODE |
| `ode_time_horizon` | 1.0 | Горизонт времени ODE |
| `ode_n_steps` | 20 | Количество шагов ODE |
| `ode_chunk_size` | 256 | Размер чанка для больших полей |
| `sde_noise_level` | 0.01 | Уровень шума SDE |
| `adjoint_enabled` | False | Adjoint method для ODE |
| `response_smoothness_target` | 0.92 | Целевая гладкость отклика |

#### Фаза 8: Агентная оркестрация

| Параметр | Default | Описание |
|----------|---------|----------|
| `agent_orchestration` | False | Включить AgentPlanner |
| `max_plan_depth` | 3 | Макс. глубина плана |
| `max_tool_calls` | 5 | Макс. вызовов инструментов |
| `tool_timeout` | 15.0 | Таймаут инструмента (сек) |
| `hypothesis_verification` | True | Верификация гипотез |
| `verification_confidence_threshold` | 0.7 | Порог уверенности верификации |
| `goal_directed_routing` | False | Целенаправленная маршрутизация |

#### Фаза 9: Продакшен-стек

| Параметр | Default | Описание |
|----------|---------|----------|
| `production_mode` | False | Режим продакшена |
| `eval_mode` | `EvalMode.PRODUCTION` | Режим оценки |
| `shadow_mode` | False | Shadow mode |
| `shadow_fallback_threshold` | 0.3 | Порог fallback shadow mode |
| `auto_rollback` | False | Автооткат |
| `auto_rollback_threshold` | 0.15 | Порог автоотката |
| `eval_frequency` | 100 | Частота оценки |
| `ragas_enabled` | False | RAGAS++ оценка |
| `drift_detection` | False | Обнаружение дрейфа |
| `drift_window` | 100 | Окно дрейфа |
| `drift_threshold` | 0.05 | Порог дрейфа |
| `metrics_retention` | 10000 | Удержание метрик |

#### Фаза 10: Кросс-модальность + Мета-контроллер + Федерация

| Параметр | Default | Описание |
|----------|---------|----------|
| `cross_modal` | False | Кросс-модальный резонанс |
| `modal_phase_offsets` | {...} | Сдвиги фаз по модальностям |
| `cross_modal_kernel_weight` | 0.35 | Вес кросс-модального ядра |
| `meta_controller` | False | MetaController (Optuna) |
| `meta_optimization_freq` | 500 | Частота мета-оптимизации |
| `meta_n_trials` | 20 | Кол-во trials Optuna |
| `meta_optimize_params` | [decay_rate, ...] | Параметры для оптимизации |
| `federated` | False | Федеративная синхронизация |
| `federated_sync_lr` | 0.01 | LR федеративной синхронизации |
| `federated_sync_freq` | 100 | Частота федеративной синхронизации |
| `federated_min_resonance` | 0.2 | Мин. резонанс для обмена |
| `node_id` | "local" | ID узла федерации |

#### Фаза 11: Стратификация + Гиперболическая геометрия + Predictive Coding + Counterfactual + DP

| Параметр | Default | Описание |
|----------|---------|----------|
| `memory_tiers` | {"episodic","semantic","procedural"} | Уровни памяти |
| `tier_decay` | {episodic:0.992, ...} | Decay rate по уровням |
| `tier_tension_thresh` | {episodic:0.10, ...} | Порог напряжения по уровням |
| `hyperbolic` | False | Геометрия Пуанкаре |
| `ball_radius` | 0.85 | Радиус шара Пуанкаре |
| `curvature` | -1.0 | Кривизна пространства |
| `predictive_coding` | False | Предсказательное кодирование |
| `pc_latent_dim` | 32 | Латентная размерность PC |
| `pc_lr` | 0.01 | Скорость обучения PC |
| `counterfactual_imagination` | False | Контрфактуальное воображение |
| `max_scenarios` | 5 | Макс. сценариев |
| `differential_privacy` | False | Дифференциальная приватность |
| `dp_epsilon` | 2.0 | Бюджет приватности |
| `dp_delta` | 1e-5 | Вероятность выхода за бюджет |
| `dp_max_norm` | 1.0 | Макс. норма обновления |

#### Фаза 12: MoE + Сжатие + Кристаллизация + Async

| Параметр | Default | Описание |
|----------|---------|----------|
| `sparse_routing` | False | MoE-маршрутизация |
| `num_shards` | 8 | Количество шардов |
| `top_shards` | 3 | Топ шардов для запроса |
| `cognitive_compression` | False | Когнитивное сжатие |
| `high_resonance_threshold` | 0.6 | Порог высокого резонанса |
| `crystallization` | False | Кристаллизация памяти |
| `crystallization_freq` | 200 | Частота кристаллизации |
| `crystallization_similarity` | 0.75 | Порог схожести |
| `crystallization_min_cluster` | 3 | Мин. размер кластера |
| `async_pipeline` | False | Async multi-threaded pipeline |
| `query_queue_size` | 50 | Размер очереди запросов |
| `save_queue_size` | 100 | Размер очереди сохранения |
| `evolve_queue_size` | 20 | Размер очереди эволюции |

#### Фаза 13: Телеология + Внимание + RL + Event-driven

| Параметр | Default | Описание |
|----------|---------|----------|
| `goal_tracking` | False | Отслеживание целей |
| `max_goals` | 20 | Макс. целей |
| `goal_decay` | 0.995 | Затухание целей |
| `goal_completion_threshold` | 0.8 | Порог выполнения цели |
| `attention_bias` | False | Когнитивное внимание |
| `bias_temperature` | 1.0 | Температура внимания |
| `rl_feedback` | False | RL из ответов LLM |
| `rl_learning_rate` | 0.01 | Скорость обучения RL |
| `rl_reward_window` | 10 | Окно награды RL |
| `event_driven` | False | Event-driven обработка |
| `low_rank_compression` | False | Low-rank SVD сжатие |
| `compression_rank` | 32 | Ранг сжатия |
| `compression_freq` | 500 | Частота сжатия |

#### Фаза 14: Мета-память + Безопасность + Рой

| Параметр | Default | Описание |
|----------|---------|----------|
| `meta_memory` | False | Introspective Meta-Memory |
| `self_reflection_freq` | 100 | Частота саморефлексии |
| `memory_age_factor` | 0.001 | Фактор возраста памяти |
| `recall_accuracy_threshold` | 0.6 | Порог точности вспоминания |
| `security_enabled` | False | Защита от инъекций |
| `max_node_text_length` | 10000 | Макс. длина текста узла |
| `tension_spike_threshold` | 0.5 | Порог всплеска напряжения |
| `causal_graph_integrity_check` | True | Проверка целостности каузального графа |
| `prompt_injection_patterns` | [...] | Шаблоны инъекций |
| `swarm_memory` | False | Роевая память |
| `swarm_consensus_threshold` | 0.5 | Порог консенсуса роя |
| `swarm_max_agents` | 10 | Макс. агентов в рое |
| `swarm_vote_weight` | 0.3 | Вес голоса агента |

#### Фаза 15: Memory Git + Clarification + Attention + Entropy + Triton

| Параметр | Default | Описание |
|----------|---------|----------|
| `version_control` | False | Delta-based versioning |
| `max_versions` | 100 | Макс. версий |
| `proactive_clarification` | False | Уточняющие вопросы при слабом резонансе |
| `clarification_threshold_ratio` | 0.5 | Порог clarification относительно min_response |
| `attention_tokens` | True | [ATTN:x][SAL:x][TIER:x] токены |
| `entropy_management` | False | Shannon entropy контроль |
| `entropy_high_threshold` | 3.0 | Порог высокого шума |
| `entropy_low_threshold` | 0.5 | Порог застоя |
| `triton_backend` | False | GPU-ускорение резонанса |
| `min_nodes_for_gpu` | 2000 | Мин. узлов для GPU |

#### Фаза 16: SymbolicOverlay + SafetyCertifier + UMP

| Параметр | Default | Описание |
|----------|---------|----------|
| `symbolic_overlay` | False | Probabilistic logic layer |
| `symbolic_min_self_sup` | 0.7 | Мин. self_sup_score для правил |
| `symbolic_max_tension` | 0.15 | Макс. tension для правил |
| `symbolic_confidence_threshold` | 0.65 | Порог уверенности вывода |
| `safety_certifier` | False | Lyapunov soft regulator |
| `safety_mode` | "soft_regulate" | monitor_only / soft_regulate / hard_block |
| `lyapunov_alpha` | 0.4 | Вес tension² в V |
| `lyapunov_beta` | 0.4 | Вес entropy в V |
| `lyapunov_gamma` | 0.2 | Вес causal_conflict в V |
| `lyapunov_threshold` | 0.1 | Порог dV/dt |
| `ump_enabled` | False | Universal Memory Protocol |

#### Фаза 17: RoleShardRouter

| Параметр | Default | Описание |
|----------|---------|----------|
| `role_sharding` | False | Ролевое шардирование |
| `role_shards` | {"default"} | Начальные шарды |
| `cross_shard_threshold` | 0.45 | Порог обмена между шардами |
| `auto_role_detection` | True | Автодетект роли по тексту |

#### Фаза 21: Self-Organizing Tokenizer + Embedding Field

| Параметр | Default | Описание |
|----------|---------|----------|
| `sot_enabled` | False | Включить SOT |
| `sot_token_dim` | None | Размерность токен-эмбеддингов (None = latent_dim) |
| `sot_max_vocab` | 4096 | Максимальный размер vocab |
| `sot_merge_threshold` | 0.7 | Порог co-retrieval для merge |
| `sot_contrastive_lr` | 0.01 | LR контрастного Хебба |
| `sot_negatives_per_query` | 5 | Число negative samples на query |
| `sot_ssm_sync` | False | SSM-синхронизация эмбеддингов |
| `sot_diagonal_ssm` | True | Диагональный SSM O(N*d) вместо O(N*d^2) |
| `sot_merge_freq` | 100 | Шагов между merge-попытками |
| `sot_min_cooccurrence` | 5 | Минимальная cooccurrence для merge |
| `sot_use_for_query` | False | Использовать SOT эмбеддинги для query (без external API) |
| `sot_tokenization_mode` | "byte" | Режим токенизации: "byte" или "word" |
| `sot_warm_start_corpus` | None | Путь к JSON корпусу для warm-start PMI |
| `sot_subword_seed` | False | Пресеять частые byte биграммы/триграммы |
| `sot_attention_pooling` | False | IDF-взвешенное pooling с бонусом позиции |
| `sot_hard_negatives` | False | Использовать ближайшие negatives |
| `sot_retrieval_feedback` | False | Обновлять эмбеддинги от результатов query |
| `sot_skipgram_window` | 1 | Окно skip-gram для co-occurrence |
| `sot_bootstrap_projection` | None | Путь к .npz файлу SBERT bootstrap |
| `sot_bootstrap_corpus` | None | Путь к корпусу для автоматического bootstrap |
| `sot_bootstrap_model` | "all-MiniLM-L6-v2" | Модель для auto-bootstrap (SBERT) |
| `sot_bootstrap_fasttext_model` | None | Путь к gensim модели для FastText bootstrap |
| `sot_max_cooccurrence` | 100_000 | Лимит записей co-occurrence |

**SOT Bootstrap (CLI):**
```bash
python -m rtmdk bootstrap corpus.json --output bootstrap.npz
```

**Word-mode + bootstrap (Python):**
```python
cfg = RTMDKConfig(
    latent_dim=64,
    sot_enabled=True,
    sot_tokenization_mode="word",
    sot_bootstrap_projection="bootstrap.npz",  # или sot_bootstrap_corpus="corpus.json"
)
field = RTMDKField(cfg)
# field.sot_tokenizer.bootstrap_from_teacher(texts, teacher_fn)  # ручной вызов
```

#### Фаза 22: Mathematical Enhancements (P0–P2)

| Параметр | Default | Описание |
|----------|---------|----------|
| `hyperbolic` | False | Геометрия Пуанкаре (P0.1) |
| `ball_radius` | 0.85 | Радиус шара Пуанкаре (P0.1) |
| `adaptive_bandwidth` | False | Локальная адаптивная ширина ядра k-NN KDE (P1.2) |
| `adaptive_bandwidth_k` | 5 | k для k-NN оценки плотности (P1.2) |
| `conformal_prediction` | False | Inductive Conformal Prediction для retrieval confidence (P1.1) |
| `conformal_alpha` | 0.10 | Уровень значимости ICP (1 − coverage) (P1.1) |
| `conformal_min_calib` | 50 | Мин. размер калибровочной выборки (P1.1) |
| `spectral_consolidation` | False | Spectral Graph Laplacian для кластеризации перед merge (P2.1) |
| `spectral_max_clusters` | 10 | Макс. число кластеров (авто-выбор через eigengap) (P2.1) |
| `spectral_sigma` | 1.0 | Ширина ядра affinity-матрицы (P2.1) |
| `enable_kalman_filter` | False | EKF неопределённости позиций узлов (P2.2) |
| `kalman_diagonal_approx` | True | Диагональное приближение ковариации (экономия памяти) (P2.2) |
| `kalman_process_noise` | 0.01 | Шум процесса Q (P2.2) |
| `kalman_measurement_noise` | 0.1 | Шум измерения R (P2.2) |
| `kalman_init_variance` | 1.0 | Начальная дисперсия ковариации (P2.2) |

---

## 5. Ядро: RTMDKMemory и RTMDKField

### RTMDKMemory — главный интерфейс

```python
memory = RTMDKMemory(config=RTMDKConfig(), embedder=Callable[[str], np.ndarray])
```

| Метод | Возвращает | Описание |
|-------|------------|----------|
| `save_context(inputs, outputs)` | None | Сохраняет текст в память (создаёт узел) |
| `load_memory_variables(inputs)` | Dict[str, str] | Извлекает top-k релевантных узлов |
| `get_stats()` | Dict | Полная статистика поля |
| `get_field_health()` | Dict | Здоровье поля |
| `get_contradictions()` | List[ContradictionRecord] | Активные противоречия |
| `get_causal_summary()` | Dict | Сводка каузального графа |
| `get_active_goals()` | List[GoalNode] | Активные цели |
| `add_goal(desc, priority)` | str (goal_id) | Добавить цель |
| `imagine_counterfactual(query, intervention)` | List[Dict] | Сценарии «что если» |
| `do_intervention(node_id, text)` | None | Каузальное вмешательство do(X) |
| `rollback(n_steps)` | bool | Откат консолидаций |
| `export_field(path)` | None | Экспорт в JSON |
| `import_field(path, embedder)` | RTMDKMemory | Импорт из JSON (classmethod) |
| `export_ump(path)` | None | Экспорт в Universal Memory Protocol |
| `import_ump(path, embedder)` | RTMDKMemory | Импорт из UMP (classmethod) |
| `validate_ump(path)` | Dict | Валидация UMP-файла |
| `clear()` | None | Очистка памяти |
| `query_by_text(text, top_k)` | List[(id, score, node)] | Query через SOT без external embedder |

### RTMDKField — внутренняя реализация

`RTMDKField` — низкоуровневый движок поля. Прямое использование редко нужно, но полезно для расширенного контроля:

```python
from rtmdk import RTMDKField, RTMDKConfig

field = RTMDKField(config)
field.add_node(embedding, {"text": "hello"}, modality="text")
# Batch ingestion (vectorized, single WAL write, single cache invalidation)
field.add_nodes_batch(embeddings_array, contents_list, modalities=["text"]*len(contents_list))
# Async background ingestion (returns immediately, worker processes in background)
field.queue_add_nodes(embeddings_array, contents_list)
results = field.query(embedding, phase=0.0, top_k=5)
field.step()  # Продвинуть динамику на 1 шаг
field.consolidate()  # Запустить консолидацию
integrity = field._check_field_integrity()  # Проверка NaN/inf

# WAL replay (automatic on RTMDKMemory startup when wal_path is set)
# Embeddings are persisted in WAL for exact recovery; fallback to embedder if missing
```

### MCP Server (Model Context Protocol)

```python
# Stdio mode (Claude Desktop, Cursor, Windsurf)
python -m rtmdk.mcp

# SSE mode (remote clients)
python -m rtmdk.mcp --transport sse --port 8080

# CLI entry point
rtmdk-mcp
```

**Tools:** `add_memory`, `query_memory`, `delete_memory`, `consolidate_memory`, `get_memory_stats`

**Resources:** `memory://stats`, `memory://nodes`, `memory://node/{id}`

**Prompts:** `memory://prompts/context` — system prompt enriched with top-k relevant memories

---

## 6. Утилиты

### modality.py

```python
from rtmdk import detect_modality, detect_tier

detect_modality("def foo(): return 42")     # → "code"
detect_modality("sample rate 44100 Hz")     # → "audio"
detect_modality("image 1920x1080")          # → "vision"
detect_modality("latency p99 = 50ms")       # → "metrics"
detect_modality("hello world")              # → "text"

detect_tier("how to install pip", {})       # → "procedural"
detect_tier("2024-01-15 meeting", {})       # → "episodic"
detect_tier("Paris is capital of France", {}) # → "semantic"
```

### hyperbolic.py

```python
from rtmdk import poincare_dist, mobius_add
import numpy as np

u = np.random.randn(64).astype(np.float32) * 0.1
v = np.random.randn(64).astype(np.float32) * 0.1
d = poincare_dist(u, v, ball_radius=0.85)
w = mobius_add(u, v, ball_radius=0.85)
```

### attention.py

```python
from rtmdk import apply_attention_bias, format_cognitive_context
biased = apply_attention_bias(results, temperature=1.0)
context = format_cognitive_context(biased)
# → ### COGNITIVE_CONTEXT
#   [SCORE:0.428][TIER:S][CAUSAL:2] coffee helps wake up
```

### formatting.py

```python
from rtmdk import format_context, build_system_prompt, ContextFormat

ctx = format_context(results, ContextFormat.ATTENTION)
# → ### ATTENTION_CONTEXT
#   [ATTN:0.428][SAL:0.598][TIER:S] coffee helps wake up
prompt = build_system_prompt(ctx, ContextFormat.ATTENTION, use_structured=True)
```

---

## 7. Движки (Engines)

### CausalInferenceEngine

Обнаружение каузальной структуры и do-calculus.

```python
engine = CausalInferenceEngine(min_samples=20, p_threshold=0.05)
engine.record_observation(["n0", "n1", "n2"], context={"session": "u1"})
parents = engine.discover_causal_structure()
prob = engine.compute_do_probability(effect="n2", intervention="n0")
result = engine.counterfactual_query(intervention={"n0": 1.0}, query_nodes=["n2", "n3"])
safety = engine.validate_consolidation("n0", "n1")
engine.do_intervention("n0", new_position)
```

### PredictiveCodingModel

Предсказание следующей конфигурации поля.

```python
pc = PredictiveCodingModel(latent_dim=64, lr=0.01)
pc.update(state_t, state_t1)
predicted = pc.predict(state_t)
fe = pc.compute_free_energy(state_t, state_t1)
```

### NeuralODEDynamics

Непрерывная эволюция поля через ODE/SDE.

```python
ode = NeuralODEDynamics(latent_dim=64, noise_level=0.01, solver="RK45")
trajectory = ode.evolve(initial_state, t_span=np.linspace(0, 1, 20))
trajectory = ode.evolve_with_noise(initial_state, dt=0.05)
grad = ode.compute_topology_gradient(nodes)
```

### DifferentialPrivacy

DP-SGD для федеративных обновлений.

```python
dp = DifferentialPrivacy(epsilon=2.0, delta=1e-5, max_norm=1.0)
clipped = dp.clip_update(gradient)
noisy = dp.add_noise(clipped)
dp.record_update()
spent = dp.get_privacy_spent()
```

### ScenarioPlanner

Генерация «что если» сценариев.

```python
planner = ScenarioPlanner(field, max_scenarios=5)
scenarios = planner.imagine_counterfactual(base_query=embedding, intervention={"n0": 0.5, "n3": -0.3})
```

---

## 8. Поддержка (Support)

### Оптимизация и адаптация

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `MetaController` | Автоподстройка гиперпараметров | `optimize(field)`, `apply_params(field, params)` |
| `MetaAdaptiveKernel` | Адаптация bandwidth/phase_coupling по куртозису | `adapt()`, `record_response()`, `get_bandwidth()` |
| `IncPCAProjection` | Learned dimensionality reduction | `update(emb)`, `project(emb)`, `set_matrix(mat)` |
| `LearnableKernel` | Differentiable resonance + Adam | `resonance_response()`, `compute_gradients()`, `step()` |

### Топология и здоровье

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `TopologyHealer` | Лечение аномалий поля | `compute_field_health()`, `heal_dead_zones()`, `detect_fragmentation()` |
| `TDAMonitor` | Топологический мониторинг | `compute_persistence()`, `get_trend()` |
| `AdaptiveThreshold` | Скользящий порог | `record_tension()`, `get_threshold()` |

### Поиск и индексация

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `HNSWIndex` | Approximate nearest neighbors | `insert(id, pos)`, `search(pos, top_k)` |
| `BM25Index` | Текстовый поиск | `add_document(id, text)`, `search(query, top_k)` |
| `TorchBackend` | GPU ускорение | `batch_resonance(...)`, `available` |

### Федерация и синхронизация

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `KuramotoSync` | Синхронизация фаз (dφ/dt = K·sin(Δφ)) | `step()`, `compute_order_parameter()`, `sync_to_target()` |
| `FederatedRTMDK` | Федеративная синхронизация | `register_peer()`, `sync_with_peers()`, `get_aggregated_params()` |

### Телеология, RL, события

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `GoalTracker` | Управление целям | `add_goal()`, `update_completion()`, `get_goal_relevance()` |
| `RLFeedbackLoop` | RL из LLM-ответов | `extract_reward_from_response()`, `apply_field_updates()` |
| `EventDrivenScheduler` | Очередь событий | `enqueue(type, payload)`, `process_pending()` |
| `LowRankCompressor` | SVD сжатие | `compress(positions)`, `get_compression_ratio()` |

### Мета-память и безопасность

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `MetaMemoryEvaluator` | Интроспекция памяти | `evaluate_recall_accuracy()`, `self_reflect()`, `get_adaptive_params()` |
| `SecurityValidator` | Защита от атак | `validate_node_content()`, `validate_tension_spike()`, `validate_causal_graph_integrity()` |

### Рой

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `SwarmConsensusProtocol` | Консенсус агентов | `register_agent()`, `propose_attractor()` |

### Phase 15: Версионирование, Энтропия, Triton

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `VersionControl` | Delta-based versioning | `create_version()`, `diff()`, `rollback_to()`, `history()` |
| `EntropyController` | Shannon entropy управление | `compute_entropy()`, `should_consolidate()`, `get_consolidation_multiplier()` |
| `TritonBackend` | GPU-accelerated resonance | `batch_resonance()`, `should_use_gpu()` |

### Phase 16: Символика, Безопасность, UMP

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `SymbolicOverlay` | Probabilistic Horn clauses | `extract_rules_from_field()`, `forward_chain()`, `get_symbolic_context()` |
| `SafetyCertifier` | Lyapunov soft regulator | `check_and_regulate()`, `get_regulation_factor()`, `should_block_updates()` |
| `UniversalMemoryProtocol` | Стандартизированный экспорт | `export()`, `import_ump()`, `validate()` |

### Phase 17: Ролевое шардирование

| Класс | Назначение | Ключевые методы |
|-------|-----------|-----------------|
| `RoleShardRouter` | Ролевая маршрутизация | `add_node()`, `get_relevant_shards()`, `update_kuramoto_phases()` |
| `RoleShard` | Один шард | `to_dict()`, `from_dict()` |
| `RoleDetector` | Автодетект роли | `detect(text)` |

---

## 9. Агенты и Продакшен

### Агенты (rtmdk.support.agents)

```python
from rtmdk import AgentPlanner, HypothesisVerifier, ToolRouter
planner = AgentPlanner(max_depth=3, max_tool_calls=5)
plan = planner.create_plan("Найти информацию о кофе", available_tools=["retrieve", "search"], context={})
verifier = HypothesisVerifier(confidence_threshold=0.7)
hyp = verifier.verify("Кофе → бодрость", causal_engine, active_nodes=["n0", "n1"])
router = ToolRouter(timeout=15.0)
router.register_tool("retrieve", lambda query: memory.load_memory_variables({"input": query}))
result = router.execute("retrieve", {"query": "кофе"})
```

### Продакшен (rtmdk.support.production)

```python
from rtmdk import ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager
shadow = ShadowModeEvaluator(fallback_threshold=0.3)
cmp = shadow.compare(shadow_output=0.8, production_output=0.75)
ragas = RAGASPlusEvaluator()
eval_result = ragas.evaluate(question="Что я пью?", answer="Кофе", contexts=["Я люблю кофе по утрам"], ground_truth="Кофе")
rollback = AutoRollbackManager(threshold=0.15)
rollback.set_baseline(0.85)
triggered = rollback.record_score(0.5)  # → True (деградация > 0.15)
```

---

## 10. EmbedderFactory

Единый интерфейс для всех типов эмбеддингов.

```python
from embedder_factory import EmbedderFactory

embedder = EmbedderFactory.create("dummy", dim=768, seed=42)  # Детерминированный
embedder = EmbedderFactory.create("lmstudio", url="http://localhost:12345/v1")  # LM Studio
embedder = EmbedderFactory.create("sentence", model="all-MiniLM-L6-v2")  # Sentence Transformers

vec = embedder("hello world")  # → np.ndarray shape (768,)
```

| Режим | Сеть | Скорость | Качество | Зависимости |
|-------|------|----------|----------|-------------|
| `dummy` | Нет | Мгновенно | Синтетическое | numpy |
| `lmstudio` | Нужен сервер | ~50ms | Высокое | requests |
| `sentence` | Нет | ~100ms | Высокое | sentence-transformers |

---

## 11. Сервер и HTTP API

### Запуск

```bash
python rtmdk_server.py  # → http://0.0.0.0:8080
# или
docker-compose up -d
```

### Переменные окружения

| Переменная | Default | Описание |
|------------|---------|----------|
| `RTMDK_HOST` | `0.0.0.0` | Хост сервера |
| `RTMDK_PORT` | `8080` | Порт сервера |
| `RTMDK_MEMORY_FILE` | `~/.rtmdk/memory.json` | Путь к файлу памяти |
| `RTMDK_ENABLE_LM_STUDIO` | `true` | Включить LM Studio интеграцию |
| `LM_STUDIO_URL` | `http://localhost:12345/v1` | URL LM Studio |
| `RTMDK_API_KEY` | `rtmdk-local` | API ключ |
| `RTMDK_AUTO_SAVE` | `60` | Интервал автосохранения (сек) |
| `RTMDK_LM_STUDIO_TIMEOUT` | `30` | Таймаут запросов к LM Studio (сек) |

### Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Статус сервера |
| GET | `/v1/models` | Список моделей |
| POST | `/v1/chat/completions` | Чат с памятью (OpenAI-compatible) |
| POST | `/v1/embeddings` | Эмбеддинги |
| GET | `/v1/memory/stats` | Статистика памяти |
| GET | `/v1/memory/health` | Здоровье поля |
| POST | `/v1/memory/imagine` | Контрфактуальные сценарии |
| POST | `/v1/memory/intervene` | Do-интервенция |
| POST | `/v1/memory/save` | Сохранить контекст |
| POST | `/v1/memory/query` | Запросить память |
| POST | `/v1/memory/export` | Экспорт в JSON |
| POST | `/v1/memory/import` | Импорт из JSON |
| POST | `/v1/memory/clear` | Очистить память |
| GET | `/v1/memory/causal` | Каузальная сводка |
| GET | `/v1/memory/contradictions` | Противоречия |

### GraphQL API

Endpoint: `POST /graphql`

| Операция | Пример |
|----------|--------|
| Query health | `{ health { status version memoryNodes } }` |
| Query node | `{ node(id: "n0") { id content salience phase amplitude } }` |
| Query nodes | `{ nodes(limit: 10, offset: 0) { id content salience } }` |
| Create node | `mutation { createNode(content: "hello", salience: 0.8) { id content } }` |
| Delete node | `mutation { deleteNode(id: "n0") }` |

### WebSocket Streaming

Endpoint: `WS /ws/memory`

```json
// Query
{"action": "query", "query": "hello", "top_k": 5}
// → {"type": "query_results", "results": [...]}

// Ping
{"action": "ping"}
// → {"type": "pong"}
```

### SOT Endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/v1/sot/status` | Статус SOT: vocab_size, max_vocab, merges_count |
| POST | `/v1/sot/bootstrap` | Bootstrap из корпуса: `{"texts": [...], "teacher_model": "..."}` |
| GET | `/v1/sot/vocab` | Словарь SOT с пагинацией: `?limit=100&offset=0&search=hello` |

---

## 12. CLI-чат с LM Studio

```bash
python lmstudio_rtmdk_chat.py
```

### Команды

| Команда | Описание |
|---------|----------|
| `/stats` | Полная статистика |
| `/tiers` | Распределение по уровням |
| `/health` | Здоровье поля |
| `/causal` | Каузальная сводка |
| `/contradict` | Противоречия |
| `/whatif {"do": {...}, "query": [...]}` | Контрфактуал |
| `/imagine {"query": "...", "intervention": {...}}` | Сценарии |
| `/hyperbolic` | Геометрия Пуанкаре |
| `/predictive` | Предсказательное кодирование |
| `/privacy` | Дифференциальная приватность |
| `/shards` | MoE-шардирование |
| `/crystallize` | Кристаллизация |
| `/compression` | Когнитивное сжатие |
| `/format json\|yaml\|plain` | Формат контекста |
| `/session <id>` | Переключить сессию |
| `/export` / `/clear` / `/quit` | Управление |

---

## 13. Тестирование

```bash
# Smoke test (быстрая проверка)
python tests/smoke_test.py

# Eval pipeline
python eval_pipeline.py --n_samples 50

# Swarm simulation
python swarm_memory.py --n_agents 5 --n_rounds 10

# Pytest (если установлен)
pip install pytest
python -m pytest test_rtmdk_v8.py -v    # 34 теста v8
python -m pytest test_rtmdk_v7.py -v    # 32 теста v7
```

### Smoke test проверяет

1. Создание памяти с async_pipeline + HNSW + attention_bias
2. save_context + async evolution
3. load_memory_variables с attention bias
4. 30 шагов консолидации без KeyError
5. HNSW routing
6. Adaptive threshold
7. Rollback history
8. Cognitive context compression
9. Attention bias

---

## 14. Обратная совместимость

### Импорт/экспорт между версиями

```python
# Экспорт из v5
from rtmdk_memory_v5 import RTMDKMemory as V5Memory
v5 = V5Memory(config=v5_config, embedder=embedder)
v5.export_field("state.json")

# Импорт в v8
from rtmdk import RTMDKMemory as V8Memory
v8 = V8Memory.import_field("state.json", embedder)
```

### Гарантии

- Все поля `MemoryNode` сохраняются/загружаются корректно
- Новые поля получают значения по умолчанию при загрузке старого state
- `RTMDKConfig` — новые параметры получают дефолтные значения
- Пакет `rtmdk/` — единственный актуальный источник кода

---

## 15. Развёртывание

### Docker Compose

```yaml
services:
  rtmdk-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - RTMDK_HOST=0.0.0.0
      - RTMDK_PORT=8080
      - RTMDK_MEMORY_FILE=/data/memory.json
    volumes:
      - rtmdk-data:/data
    restart: unless-stopped
volumes:
  rtmdk-data:
    driver: local
```

### IDE интеграция

**Cursor / Continue / Aider:**
```json
{
  "apiBaseUrl": "http://localhost:8080/v1",
  "apiKey": "rtmdk-local"
}
```

**OpenAI SDK (Python):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="rtmdk-local")
response = client.chat.completions.create(
    model="rtmdk",
    messages=[{"role": "user", "content": "Что я говорил вчера?"}],
    session_id="user1"
)
```

### Streamlit Dashboard

```bash
pip install streamlit matplotlib pandas
streamlit run streamlit_app.py
# → http://localhost:8501
```

---

## 16. История коммитов

| Коммит | Описание | Файлов | Строк |
|--------|----------|--------|-------|
| `8c7747b` | Модуляризация + фиксы стабильности | 44 | +3846 |
| `0119af1` | DOCUMENTATION.md + README | 2 | +900 |
| `98c49d0` | Phase 15 (Memory Git, Clarification, Attention, Entropy, Triton) | 9 | +814 |
| `66e0370` | LOCAL_SETUP.md | 1 | +340 |
| `40e63ff` | Phase 16 (SymbolicOverlay, SafetyCertifier, UMP v1) | 7 | +1087 |
| `4391b1c` | Phase 17 (RoleShardRouter) | 5 | +368 |
| `e8ccfb1` | Аудит: 10 критических фиксов (performance, stability, LM Studio) | 3 | +313/-148 |
| `6f7b781` | Sync rtmdk/ модулей с монолитом | 3 | +54 |

*Документация актуальна для коммита `6f7b781`.*

---

## 17. Audit Log

### GET /v1/admin/audit-log

Query audit log entries (admin only).

**Query parameters:**
- `actor` (optional): Filter by actor identifier
- `action` (optional): Filter by action type (e.g., `create_node`, `update_node`, `delete_node`)
- `since` (optional): Unix timestamp — only entries after this time
- `limit` (default 100, max 1000): Maximum number of entries to return

**Headers:**
- `X-API-Key: <admin_key>`

**Response:**
```json
{
  "entries": [
    {
      "timestamp": 1715078400.123,
      "action": "create_node",
      "actor": "tenant_1",
      "resource": "node_abc123",
      "details": {"content_preview": "Hello world"}
    }
  ],
  "count": 1
}
```

---

## 18. Data Retention

### GET /v1/admin/retention

Get retention manager statistics (admin only).

**Headers:**
- `X-API-Key: <admin_key>`

**Response:**
```json
{
  "pruned_total": 42,
  "policy": {
    "enabled": true,
    "max_age_seconds": 2592000,
    "max_nodes": null
  }
}
```

**Environment variables:**
- `RTMDK_RETENTION_MAX_AGE_DAYS` — automatic pruning of nodes older than N days (0 = disabled)
- `RTMDK_RETENTION_MAX_NODES` — keep only N most recently accessed nodes (0 = disabled)

---

## 14. Pipeline API (v8.3+)

Явный retrieval pipeline с 6 стадиями, каждая из которых независимо наблюдаема.

### Python API

```python
from rtmdk import RTMDKMemory, RTMDKConfig

config = RTMDKConfig.production()
mem = RTMDKMemory(config=config, embedder=embed_fn)

result = mem.retrieve_nodes_pipeline("What is resonance?", top_k=5)
# result["results"]      — ranked nodes: [(node_id, score, node), ...]
# result["route"]        — routing decision: "factual" | "standard" | "deep"
# result["explanations"] — per-result explanation dicts
# result["metrics"]      — {stages: [...], total_latency_ms: ..., breaker_states: {...}}
```

### Batch execution

```python
from rtmdk.pipeline import BatchPipelineExecutor

batch = BatchPipelineExecutor(mem.build_pipeline().stages)
outputs = batch.run_batch(["q1", "q2", "q3"], top_k=5)
# outputs — list of ctx.to_dict()
```

### HTTP endpoints

#### Synchronous query
```bash
curl -X POST http://localhost:8080/v1/memory/query_pipeline \
  -H "Content-Type: application/json" \
  -d '{"query": "resonance", "top_k": 5, "session_id": "sess_1"}'
```

#### SSE streaming (live stage events)
```bash
curl -N 'http://localhost:8080/v1/memory/pipeline/stream?query=resonance&top_k=5'
```

**Events:**
- `pipeline_started` — planned stage list
- `stage_started` — stage execution begins
- `stage_completed` — stage done with latency and breaker state
- `stage_degraded` — stage failed or bypassed
- `pipeline_completed` — final results and total latency

#### Query plan preview
```bash
curl 'http://localhost:8080/v1/memory/pipeline/plan?query=hello&route=fast&top_k=5'
```

**Response:**
```json
{
  "query": "hello",
  "plan": {
    "stage_names": ["embed", "route", "retrieve"],
    "skipped_stages": ["rerank", "calibrate", "explain"],
    "estimated_latency_ms": 16.1,
    "estimated_cost": 0.36
  }
}
```

#### Pipeline DAG
```bash
curl http://localhost:8080/v1/memory/pipeline/dag
```

**Response:**
```json
{
  "nodes": [
    {"id": "embed", "enabled": true, "has_breaker": true, "breaker_state": "closed"}
  ],
  "edges": [{"from": "embed", "to": "route"}],
  "total_stages": 6,
  "enabled_stages": 6,
  "stages_with_breakers": 6
}
```

#### Prometheus metrics
```bash
curl http://localhost:8080/v1/memory/pipeline/prometheus
```

#### Health check
```bash
curl http://localhost:8080/v1/memory/pipeline/health
```

**Response:**
```json
{
  "overall": "healthy",
  "stages": [
    {"name": "embed", "enabled": true, "breaker_state": "closed", "has_fallback": true}
  ],
  "open_breakers": 0,
  "total_stages": 6
}
```

**Response:**
```json
{
  "query": "resonance",
  "results": [{"id": "n0", "content": "...", "score": 0.95}],
  "route": "factual",
  "explanations": [],
  "metrics": {
    "stages": [
      {"stage": "embed", "latency_ms": 12.5, "error": null, "degraded": false},
      {"stage": "retrieve", "latency_ms": 0.8, "error": null, "degraded": false}
    ],
    "total_latency_ms": 13.3,
    "breaker_states": {"embed": "closed", "retrieve": "closed"}
  },
  "total": 1,
  "cost": {
    "total_cost": 0.99,
    "stage_costs": {"embed": 0.52, "retrieve": 0.02},
    "total_latency_ms": 131.0
  }
}
```

*Cost tracking доступен при `pipeline_cost_tracking_enabled=True`.*

### Query Planner

Динамическая оптимизация pipeline — пропускает ненужные стадии в зависимости от query:

```python
config = RTMDKConfig(
    pipeline_planner_enabled=True,      # Включить query planner
    pipeline_cost_tracking_enabled=True, # Включить cost tracking
)
```

**Правила оптимизации:**
- Fast route → пропускает `rerank` + `calibrate` (~40% экономия)
- Short query (<10 токенов) → пропускает `explain` (~15% экономия)
- Low top_k (≤3) → пропускает `calibrate`

### Circuit breaker configuration

```python
config = RTMDKConfig(
    pipeline_breaker_enabled=True,
    pipeline_breaker_failure_threshold=5,
    pipeline_breaker_latency_violation_threshold=3,
    pipeline_breaker_recovery_timeout_ms=30000,
    pipeline_breaker_thresholds={
        "embed": 5000.0,
        "route": 100.0,
        "retrieve": 500.0,
        "rerank": 1000.0,
        "calibrate": 200.0,
        "explain": 100.0,
    },
)
```

### Plugin registry

```python
from rtmdk.pipeline.registry import StageRegistry, GLOBAL_REGISTRY
from rtmdk.pipeline.base import PipelineStage

class CustomStage(PipelineStage):
    name = "custom"
    def process(self, ctx):
        return ctx

GLOBAL_REGISTRY.register("custom", CustomStage)
stage = GLOBAL_REGISTRY.create("custom")
```

---

*Updated: 2026-05-09 — Pipeline v8.3: Query Planner, Cost Tracking, Production Observability*
