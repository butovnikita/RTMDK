# RTMDK — Полная документация

> Версия 8.0 | Модульная архитектура | 52 публичных класса/функции | 29 файлов пакета

---

## Оглавление

1. [Обзор архитектуры](#1-обзор-архитектуры)
2. [Структура пакета](#2-структура-пакета)
3. [Установка и быстрый старт](#3-установка-и-быстрый-старт)
4. [Конфигурация](#4-конфигурация)
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

### Два стиля импорта

```python
# Модульный стиль (рекомендуется для новых проектов)
from rtmdk import RTMDKMemory, RTMDKConfig, MemoryNode

# Монолитный стиль (обратная совместимость)
from rtmdk_memory_v8 import RTMDKMemory, RTMDKConfig, MemoryNode
```

Оба стиля полностью эквивалентны. Модульный стиль даёт доступ к отдельным компонентам без загрузки тяжёлого ядра.

---

## 2. Структура пакета

```
rtmdk/                          # Главный пакет (52 публичных символа)
├── __init__.py                 # Re-export всех символов
├── config.py                   # RTMDKConfig + 5 enums
├── nodes.py                    # 10 data-классов
│
├── utils/                      # Утилиты
│   ├── __init__.py
│   ├── modality.py             # detect_modality, detect_tier
│   ├── hyperbolic.py           # poincare_dist, exp/log_map, mobius_add
│   ├── attention.py            # apply_attention_bias, format_cognitive_context
│   └── formatting.py           # format_context, build_system_prompt
│
├── engines/                    # Движки (вычисления)
│   ├── __init__.py
│   ├── causal.py               # CausalInferenceEngine
│   ├── predictive.py           # PredictiveCodingModel
│   ├── counterfactual.py       # ScenarioPlanner
│   ├── privacy.py              # DifferentialPrivacy
│   └── neural_ode.py           # NeuralODEDynamics
│
└── support/                    # Поддержка (24 класса)
    ├── __init__.py
    ├── meta_controller.py      # MetaController (Optuna/grid search)
    ├── kuramoto.py             # KuramotoSync, FederatedRTMDK
    ├── meta_adaptive.py        # MetaAdaptiveKernel
    ├── healer.py               # TopologyHealer
    ├── projection.py           # IncPCAProjection
    ├── bm25.py                 # BM25Index
    ├── threshold.py            # AdaptiveThreshold
    ├── tda.py                  # TDAMonitor
    ├── hnsw.py                 # HNSWIndex
    ├── torch_backend.py        # TorchBackend
    ├── learnable.py            # LearnableKernel, DifferentiableConsolidation
    ├── goal_tracker.py         # GoalTracker
    ├── rl_feedback.py          # RLFeedbackLoop
    ├── event_driven.py         # LowRankCompressor, EventDrivenScheduler
    ├── meta_memory.py          # MetaMemoryEvaluator
    ├── security.py             # SecurityValidator
    ├── swarm.py                # SwarmConsensusProtocol
    ├── agents/__init__.py      # AgentPlanner, HypothesisVerifier, ToolRouter
    └── production/__init__.py  # ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager
```

**Внешние файлы** (не входят в пакет `rtmdk/`):

| Файл | Описание |
|------|----------|
| `rtmdk_memory_v8.py` | Монолитное ядро: RTMDKField + RTMDKMemory (~5000 строк) |
| `rtmdk_server.py` | OpenAI-compatible HTTP-сервер |
| `lmstudio_rtmdk_chat.py` | CLI-чат через LM Studio |
| `streamlit_app.py` | Интерактивный дашборд |
| `eval_pipeline.py` | Benchmarks: ContinualQA, LongBench, MemoryBench |
| `swarm_memory.py` | Мультиагентная роевая память |
| `smoke_test.py` | Быстрая проверка критических путей |
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
pip install pytest streamlit matplotlib pandas optuna scikit-learn
# или одной командой:
pip install -r requirements-dev.txt
```

### Минимальный пример

```python
from rtmdk import RTMDKConfig, RTMDKMemory
import numpy as np

# 1. Создаём эмбеддер (заглушка для демо)
def embedder(text: str) -> np.ndarray:
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(768).astype(np.float32) * 0.1

# 2. Конфигурация с нужными фазами
config = RTMDKConfig(
    embedding_dim=768, latent_dim=64, top_k=3,
    causal_topological=True,    # Каузальные связи
    self_healing=True,          # Самовосстановление
    goal_tracking=True,         # Цели
    attention_bias=True,        # Когнитивное внимание
)

# 3. Создаём память
memory = RTMDKMemory(config=config, embedder=embedder)

# 4. Сохраняем
memory.save_context(
    {"input": "Меня зовут Никита, я разработчик", "session_id": "u1"},
    {"output": "Запомнил: Никита — разработчик"}
)

# 5. Ищем
ctx = memory.load_memory_variables({"input": "Кто я?", "session_id": "u1"})
print(ctx["rtmdk_context"])
# → [R:0.42|S:0.60] Меня зовут Никита, я разработчик
```

### С реальным эмбеддером

```python
from embedder_factory import EmbedderFactory

# LM Studio (нужен запущенный сервер на :12345)
embedder = EmbedderFactory.create("lmstudio", url="http://localhost:12345/v1")

# Sentence Transformers (локально, нужна установка)
embedder = EmbedderFactory.create("sentence", model="all-MiniLM-L6-v2")

# Dummy (детерминированный, для тестов)
embedder = EmbedderFactory.create("dummy", dim=768)
```

---

## 4. Конфигурация

### Enums

| Enum | Значения | Назначение |
|------|----------|------------|
| `ConsolidationMode` | `DIALECTICAL`, `MERGE`, `PRUNE` | Стратегия объединения узлов |
| `Backend` | `NUMPY`, `TORCH` | Бэкенд вычислений |
| `ContextFormat` | `PLAIN`, `JSON`, `YAML` | Формат контекста для LLM |
| `FieldHealth` | `STABLE`, `DEGRADED`, `CRITICAL`, `HEALING` | Состояние поля |
| `EvalMode` | `PRODUCTION`, `SHADOW`, `EVALUATION` | Режим оценки |

### RTMDKConfig — ключевые параметры по фазам

#### Базовые (фаза 0)

| Параметр | Default | Описание |
|----------|---------|----------|
| `embedding_dim` | 768 | Размерность эмбеддинга |
| `latent_dim` | 64 | Размерность скрытого многообразия |
| `top_k` | 5 | Количество узлов в ответе |
| `decay_rate` | 0.998 | Скорость затухания салентности |
| `tension_threshold` | 0.25 | Порог консолидации |
| `min_response` | 0.1 | Минимальный резонанс для выдачи |

#### Фаза 1–2: Контекст и адаптивность

| Параметр | Default | Описание |
|----------|---------|----------|
| `context_format` | `ContextFormat.PLAIN` | Формат контекста |
| `adaptive_threshold` | False | Скользящий порог консолидации |
| `soft_gates` | False | Sigmoid-ворота для узлов |
| `bm25_fallback` | False | Текстовый поиск при промахе векторного |

#### Фаза 3: Каузальность

| Параметр | Default | Описание |
|----------|---------|----------|
| `causal_topological` | False | Включить CausalInferenceEngine |
| `do_calculus_validation` | True | Валидация через do-calculus |
| `contradiction_detection` | True | Обнаружение противоречий |
| `use_hnsw` | False | Approximate nearest neighbor индекс |

#### Фаза 5: Мета-адаптивность

| Параметр | Default | Описание |
|----------|---------|----------|
| `meta_adaptive` | False | MetaAdaptiveKernel (адаптация bandwidth/phase_coupling) |
| `self_healing` | False | TopologyHealer (лечение мёртвых зон) |
| `healing_strength` | 0.1 | Сила перемещения при лечении |

#### Фаза 6: Каузально-топологическая память

| Параметр | Default | Описание |
|----------|---------|----------|
| `counterfactual_enabled` | False | Контрфактуальные запросы |
| `counterfactual_max_depth` | 3 | Максимальная глубина рассуждений |

#### Фаза 7: Neural ODE/SDE

| Параметр | Default | Описание |
|----------|---------|----------|
| `continuous_dynamics` | False | Непрерывная эволюция через ODE |
| `ode_solver` | "RK45" | Метод решения (RK45 / odeint) |
| `ode_chunk_size` | 256 | Размер чанка для больших полей |

#### Фаза 11: Стратификация + Гиперболическая геометрия

| Параметр | Default | Описание |
|----------|---------|----------|
| `memory_tiers` | {"episodic","semantic","procedural"} | Уровни памяти |
| `hyperbolic` | False | Геометрия Пуанкаре |
| `ball_radius` | 0.85 | Радиус шара Пуанкаре |
| `predictive_coding` | False | Предсказательное кодирование |
| `differential_privacy` | False | Дифференциальная приватность |
| `dp_epsilon` | 2.0 | Бюджет приватности |

#### Фаза 12: MoE + Сжатие

| Параметр | Default | Описание |
|----------|---------|----------|
| `sparse_routing` | False | MoE-маршрутизация (шардирование) |
| `num_shards` | 8 | Количество шардов |
| `cognitive_compression` | False | Когнитивное сжатие контекста |
| `crystallization` | False | Кристаллизация эпизодических → семантические |

#### Фаза 13: Телеология + Внимание + RL

| Параметр | Default | Описание |
|----------|---------|----------|
| `goal_tracking` | False | Отслеживание целей |
| `attention_bias` | False | Когнитивное внимание |
| `rl_feedback` | False | RL из ответов LLM |
| `event_driven` | False | Event-driven обработка |

#### Фаза 14: Мета-память + Безопасность

| Параметр | Default | Описание |
|----------|---------|----------|
| `meta_memory` | False | Introspective Meta-Memory |
| `security_enabled` | False | Защита от инъекций |
| `swarm_memory` | False | Роевая память |

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
| `clear()` | None | Очистка памяти |

### RTMDKField — внутренняя реализация

`RTMDKField` — низкоуровневый движок поля. Прямое использование редко нужно, но полезно для расширенного контроля:

```python
from rtmdk import RTMDKField, RTMDKConfig

field = RTMDKField(config)
field.add_node(embedding, {"text": "hello"}, modality="text")
results = field.query(embedding, phase=0.0, top_k=5)
field.step()  # Продвинуть динамику на 1 шаг
field.consolidate()  # Запустить консолидацию
```

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

# Гиперболическое расстояние (всегда ≥ евклидова)
d = poincare_dist(u, v, ball_radius=0.85)

# Мёбиусово сложение (некоммутативно, неассоциативно)
w = mobius_add(u, v, ball_radius=0.85)
```

### attention.py

```python
from rtmdk import apply_attention_bias, format_cognitive_context

# results = field.query(...) — List[(node_id, score, MemoryNode)]
biased = apply_attention_bias(results, temperature=1.0)
context = format_cognitive_context(biased)
# → ### COGNITIVE_CONTEXT
#   [SCORE:0.428][TIER:S] coffee helps wake up
#   [SCORE:0.287][TIER:S][CAUSAL:2] morning routine
```

### formatting.py

```python
from rtmdk import format_context, build_system_prompt, ContextFormat

# JSON-формат
ctx_json = format_context(results, ContextFormat.JSON)
# → [{"resonance": 0.428, "salience": 0.6, "text": "...", ...}]

# Системный промпт
prompt = build_system_prompt(ctx_json, ContextFormat.JSON, use_structured=True)
```

---

## 7. Движки (Engines)

### CausalInferenceEngine

Обнаружение каузальной структуры и do-calculus.

```python
from rtmdk import CausalInferenceEngine

engine = CausalInferenceEngine(min_samples=20, p_threshold=0.05)

# Сбор данных
engine.record_observation(["n0", "n1", "n2"], context={"session": "u1"})
engine.record_cooccurrence("n0", "n1")

# Открытие структуры
parents = engine.discover_causal_structure()

# do-calculus
prob = engine.compute_do_probability(effect="n2", intervention="n0")

# Контрфактуал
result = engine.counterfactual_query(
    intervention={"n0": 1.0},
    query_nodes=["n2", "n3"]
)

# Проверка консолидации
safety = engine.validate_consolidation("n0", "n1")
# → {"safe": True, "reasons": [], "recommendation": "proceed"}

# Вмешательство
engine.do_intervention("n0", new_position)
```

### PredictiveCodingModel

Предсказание следующей конфигурации поля.

```python
from rtmdk import PredictiveCodingModel

pc = PredictiveCodingModel(latent_dim=64, lr=0.01)

# Обучение на паре состояний
pc.update(state_t, state_t1)

# Предсказание
predicted = pc.predict(state_t)

# Свободная энергия (ошибка + сложность)
fe = pc.compute_free_energy(state_t, state_t1)
```

### NeuralODEDynamics

Непрерывная эволюция поля через ODE/SDE.

```python
from rtmdk import NeuralODEDynamics

ode = NeuralODEDynamics(latent_dim=64, noise_level=0.01, solver="RK45")

# Эволюция (ODE)
trajectory = ode.evolve(initial_state, t_span=np.linspace(0, 1, 20))

# Эволюция с шумом (SDE)
trajectory = ode.evolve_with_noise(initial_state, dt=0.05)

# Градиент топологии
grad = ode.compute_topology_gradient(nodes)
```

### DifferentialPrivacy

DP-SGD для федеративных обновлений.

```python
from rtmdk import DifferentialPrivacy

dp = DifferentialPrivacy(epsilon=2.0, delta=1e-5, max_norm=1.0)

# Обрезка + шум
clipped = dp.clip_update(gradient)
noisy = dp.add_noise(clipped)

# Отслеживание бюджета
dp.record_update()
spent = dp.get_privacy_spent()
```

### ScenarioPlanner

Генерация «что если» сценариев.

```python
planner = ScenarioPlanner(field, max_scenarios=5)
scenarios = planner.imagine_counterfactual(
    base_query=embedding,
    intervention={"n0": 0.5, "n3": -0.3}
)
# → [{"hypothetical": True, "node_id": "n0", "confidence": 0.78, "trajectory": [...]}, ...]
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

---

## 9. Агенты и Продакшен

### Агенты (rtmdk.support.agents)

```python
from rtmdk import AgentPlanner, HypothesisVerifier, ToolRouter

# Планирование
planner = AgentPlanner(max_depth=3, max_tool_calls=5)
plan = planner.create_plan("Найти информацию о кофе", available_tools=["retrieve", "search"], context={})
# → AgentPlan(goal=..., subtasks=[...], tools_needed=["retrieve"], confidence=0.65)

# Верификация
verifier = HypothesisVerifier(confidence_threshold=0.7)
hyp = verifier.verify("Кофе → бодрость", causal_engine, active_nodes=["n0", "n1"])

# Маршрутизация инструментов
router = ToolRouter(timeout=15.0)
router.register_tool("retrieve", lambda query: memory.load_memory_variables({"input": query}))
result = router.execute("retrieve", {"query": "кофе"})
```

### Продакшен (rtmdk.support.production)

```python
from rtmdk import ShadowModeEvaluator, RAGASPlusEvaluator, AutoRollbackManager

# Shadow mode
shadow = ShadowModeEvaluator(fallback_threshold=0.3)
cmp = shadow.compare(shadow_output=0.8, production_output=0.75)
# → {"difference": 0.05, "fallback_triggered": False, ...}

# RAGAS++ оценка
ragas = RAGASPlusEvaluator()
eval_result = ragas.evaluate(
    question="Что я пью?",
    answer="Кофе",
    contexts=["Я люблю кофе по утрам"],
    ground_truth="Кофе"
)
# → EvalResult(context_precision=1.0, context_recall=1.0, overall_score=0.85, ...)

# Автооткат
rollback = AutoRollbackManager(threshold=0.15)
rollback.set_baseline(0.85)
triggered = rollback.record_score(0.5)  # → True (деградация > 0.15)
```

---

## 10. EmbedderFactory

Единый интерфейс для всех типов эмбеддингов.

```python
from embedder_factory import EmbedderFactory

# Dummy — детерминированный, без сети (для тестов)
embedder = EmbedderFactory.create("dummy", dim=768, seed=42)

# LM Studio — реальные эмбеддинги через API
embedder = EmbedderFactory.create(
    "lmstudio",
    url="http://localhost:12345/v1",
    model="nomic-ai/nomic-embed-text-v1.5-GGUF"
)

# Sentence Transformers — локальная модель
embedder = EmbedderFactory.create("sentence", model="all-MiniLM-L6-v2")

# Использование
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
# Локально
python rtmdk_server.py
# → http://0.0.0.0:8080

# Docker
docker-compose up -d
```

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

### Пример: чат через API

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rtmdk","messages":[{"role":"user","content":"Привет!"}],"session_id":"u1"}'
```

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
| `/format json\|yaml\|plain` | Формат контекста |
| `/session <id>` | Переключить сессию |
| `/export` / `/clear` / `/quit` | Управление |

---

## 13. Тестирование

```bash
# Smoke test (быстрая проверка)
python smoke_test.py

# Eval pipeline
python eval_pipeline.py --n_samples 50

# Swarm simulation
python swarm_memory.py --n_agents 5 --n_rounds 10

# Pytest (если установлен)
pip install pytest
python -m pytest test_rtmdk_v8.py -v    # 34 теста v8
python -m pytest test_rtmdk_v7.py -v    # 32 теста v7
# ... и так далее для всех версий
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

---

## 15. Развёртывание

### Docker Compose

```yaml
# docker-compose.yml
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

Вкладки: Chat, Field Visualization, Goals, Security Monitor, Node Management.

---

## Приложение: Полная карта зависимостей

```
rtmdk_memory_v8.py (ядро)
  ├── rtmdk.config        → RTMDKConfig, enums
  ├── rtmdk.nodes         → MemoryNode, CausalEdge, ...
  ├── rtmdk.utils.*       → modality, hyperbolic, attention, formatting
  ├── rtmdk.engines.*     → causal, predictive, counterfactual, privacy, neural_ode
  └── rtmdk.support.*     → 24 класса поддержки

rtmdk_server.py
  └── rtmdk_memory_v8.py  → RTMDKConfig, RTMDKMemory

lmstudio_rtmdk_chat.py
  └── rtmdk_memory_v8.py  → RTMDKConfig, RTMDKMemory

streamlit_app.py
  └── rtmdk_memory_v8.py  → RTMDKConfig, RTMDKMemory, utils

eval_pipeline.py
  └── rtmdk_memory_v8.py  → RTMDKConfig, RTMDKMemory, attention

swarm_memory.py
  └── rtmdk_memory_v8.py  → RTMDKConfig, RTMDKMemory, SwarmConsensusProtocol

smoke_test.py
  └── rtmdk_memory_v8.py  → RTMDKConfig, RTMDKMemory, attention, formatting
```

---

*Документация актуальна для коммита `8c7747b` — «feat: modularize RTMDK v8 + fix critical stability issues».*
