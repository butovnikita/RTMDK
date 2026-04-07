# RTMDK — Resonance-Topological Memory

> Версия 8.0 | 326 тестов + smoke test | 8 версий с полной обратной совместимостью | OpenAI-compatible API | Docker Compose

## Обзор

RTMDK — система резонансно-топологической памяти для LLM, которая превращает пассивное хранилище в **адаптивное когнитивное поле**. Память организована как динамическое многообразие, где узлы эволюционируют через резонанс, консолидацию и непрерывную динамику.

## Быстрый старт

### Вариант 1: Docker Compose (рекомендуется)

```bash
# Запуск сервера
docker-compose up -d

# Проверка
curl http://localhost:8080/health

# Интеграция с IDE (Cursor, Continue, Aider)
# Base URL: http://localhost:8080/v1
# API Key: rtmdk-local

# Остановка
docker-compose down
```

### Вариант 2: Python (локально)

```bash
# Установка зависимостей
pip install fastapi uvicorn numpy scipy pydantic requests

# Запуск сервера
python rtmdk_server.py

# Сервер запущен на http://localhost:8080
```

### Вариант 3: Python библиотека

```python
from rtmdk_memory_v8 import RTMDKConfig, RTMDKMemory
import numpy as np

def embedder(text: str) -> np.ndarray:
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(768).astype(np.float32) * 0.1

config = RTMDKConfig(
    embedding_dim=768, latent_dim=64, top_k=5, enable_async=False,
    causal_topological=True,      # Каузальные связи
    meta_adaptive=True,           # Адаптивное ядро
    self_healing=True,            # Самовосстановление
    cross_modal=True,             # Кросс-модальность
    memory_tiers={"episodic", "semantic", "procedural"},  # Стратификация
    hyperbolic=True,              # Гиперболическая геометрия
    predictive_coding=True,       # Предсказательное кодирование
    differential_privacy=True,    # Дифференциальная приватность
    goal_tracking=True,           # Телеологический слой
    attention_bias=True,          # Когнитивное внимание
    rl_feedback=True,             # Замкнутый цикл RL
    sparse_routing=True,          # MoE-маршрутизация
    crystallization=True,         # Кристаллизация памяти
    event_driven=True,            # Event-driven архитектура
)

memory = RTMDKMemory(config=config, embedder=embedder)
memory.save_context(
    {"input": "Я люблю кофе по утрам", "session_id": "user1"},
    {"output": "Кофе помогает проснуться"}
)

ctx = memory.load_memory_variables({"input": "Что я пью по утрам?", "session_id": "user1"})
print(ctx["rtmdk_context"])

# Контрфактуальный запрос
result = memory.imagine_counterfactual(
    base_query="Что если я перейду на чай?",
    intervention={"n0": 0.5}
)

# Телеологический слой: добавить цель
memory.add_goal("Узнать предпочтения пользователя", priority=1.0)

# RL обратная связь
memory.apply_rl_feedback("Кофе — отличный выбор!", ["n0"])

# Экспорт/Импорт
memory.export_field("memory_state.json")
memory2 = RTMDKMemory.import_field("memory_state.json", embedder)
```

### Вариант 4: Smoke Test

```bash
python smoke_test.py
# Проверяет: async_pipeline, HNSW, attention_bias, consolidation, cognitive context
```

## Фазы развития

### Фаза 1: Структурированный контекст
- JSON/YAML/plain форматирование контекста
- Системные промпты с инструкциями для LLM

### Фаза 2: Адаптивные пороги
- `AdaptiveThreshold`: скользящее окно + std
- Мягкие ворота консолидации (sigmoid gate)

### Фаза 3: Каузальность + Продакшен
- `CausalInferenceEngine`: PC-algorithm, do-calculus
- `ProductionMonitor`: drift detection, anomaly detection, A/B testing

### Фаза 5: Мета-адаптивность + Самовосстановление
- `MetaAdaptiveKernel`: адаптация bandwidth/phase_coupling по куртозису
- `TopologyHealer`: обнаружение и лечение мёртвых зон, гиперконвергенции, фрагментации

### Фаза 6: Каузально-топологическая память
- P(Y|do(X)) через backdoor adjustment
- Контрфактуальные запросы
- Валидация консолидации через do-calculus
- Обнаружение противоречий: do(A)→B vs do(C)→B

### Фаза 7: Neural ODE/SDE
- `NeuralODEDynamics`: dX/dt = F(X, u) + σ·dW
- RK45 solver с fallback на odeint
- Chunking для больших N (>256 узлов)
- Response smoothness tracking

### Фаза 8: Агентная оркестрация
- `AgentPlanner`: декомпозиция целей → подзадачи → инструменты
- `HypothesisVerifier`: проверка через do-calculus
- `ToolRouter`: регистрация, выполнение, misuse rate

### Фаза 9: Продакшен-стек
- `ShadowModeEvaluator`: параллельная оценка с fallback
- `RAGASPlusEvaluator`: 6 метрик (precision, recall, relevance, faithfulness, causal, temporal)
- `AutoRollbackManager`: автоматический откат при деградации

### Фаза 10: Кросс-модальность + Мета-контроллер + Федерация
- `detect_modality()`: text/code/audio/vision/metrics
- `cross_modal_resonance()`: exp(-Δφ/π) × weight
- `MetaController`: Optuna (если доступен) или grid search
- `FederatedRTMDK` + `KuramotoSync`: dφ/dt = K·sin(φ_j - φ_i)

### Фаза 11: Стратификация + Гиперболическая геометрия + Predictive Coding + Counterfactual + DP
- `detect_tier()`: episodic/semantic/procedural
- Tier-specific decay: episodic=0.992, semantic=0.999, procedural=1.0
- `poincare_dist()`, `mobius_add()`, exp/log maps
- `PredictiveCodingModel`: free energy, surprise_level
- `ScenarioPlanner`: imagine_counterfactual, simulate_trajectory
- `DifferentialPrivacy`: clip + Gaussian noise, ε ≤ 2.0

### Фаза 12: MoE-память + Когнитивное сжатие + Кристаллизация + Async Pipeline
- **Sparse resonant routing (MoE-memory)**: шардирование с KMeans, O(N/K) вместо O(N)
- **Cognitive context compression**: `_cognitive_compress()` — структурированный дамп для LLM
- **Crystallization**: DBSCAN кластеризация эпизодических → семантические узлы
- **Async multi-threaded pipeline**: `query_q`, `save_q`, `evolve_q` с воркерами

### Фаза 13: Телеология + Внимание + RL + Event-driven
- **Teleological layer (Goal/Intent Tracking)**: `GoalTracker`, `GoalNode`, `add_goal()`, `update_goal_completion()`
- **Cognitive attention bias**: `apply_attention_bias()` — control-токены `[SCORE:x][TIER:x][CAUSAL:x][TENSION:x]`
- **Closed-loop RL feedback**: `RLFeedbackLoop` — извлечение уверенности из ответов LLM
- **Event-driven + Low-Rank compression**: `EventDrivenScheduler`, `LowRankCompressor` (инкрементальный SVD)

## Конфигурация

Все параметры через `RTMDKConfig`:

```python
config = RTMDKConfig(
    # Базовые
    embedding_dim=768, latent_dim=64, top_k=5, min_response=0.1,
    resonance_kernel="gaussian_phase",  # gaussian | cosine | gaussian_phase
    phase_coupling=0.3, bandwidth=1.0,
    attraction_lr=0.02, phase_sync_lr=0.01, decay_rate=0.998,

    # Фаза 1: Контекст
    context_format=ContextFormat.PLAIN,  # PLAIN | JSON | YAML
    use_structured_prompt=True,

    # Фаза 2: Адаптивность
    adaptive_threshold=False, soft_gates=False, gate_temperature=0.15,

    # Фаза 3: Каузальность
    causal_topological=False, do_calculus_validation=True,

    # Фаза 5: Мета-адаптивность
    meta_adaptive=False, self_healing=False,

    # Фаза 7: ODE
    continuous_dynamics=False, ode_solver="RK45",

    # Фаза 8: Агент
    agent_orchestration=False, max_plan_depth=3, max_tool_calls=5,

    # Фаза 9: Продакшен
    production_mode=False, shadow_mode=False, ragas_enabled=False,

    # Фаза 10: Кросс-модальность
    cross_modal=False, meta_controller=False, federated=False,

    # Фаза 11: Стратификация + Гиперболическая + Predictive + Counterfactual + DP
    memory_tiers={"episodic", "semantic", "procedural"},
    hyperbolic=False, ball_radius=0.85,
    predictive_coding=False, counterfactual_imagination=False,
    differential_privacy=False, dp_epsilon=2.0,

    # Фаза 12: MoE + Сжатие + Кристаллизация + Async
    sparse_routing=False, num_shards=8, top_shards=3,
    cognitive_compression=False, high_resonance_threshold=0.6,
    crystallization=False, crystallization_freq=200,
    async_pipeline=False, query_queue_size=50,

    # Фаза 13: Телеология + Внимание + RL + Event-driven
    goal_tracking=False, max_goals=20,
    attention_bias=False, bias_temperature=1.0,
    rl_feedback=False, rl_learning_rate=0.01,
    event_driven=False, low_rank_compression=False, compression_rank=32,
)
```

## API RTMDKMemory

| Метод | Описание |
|-------|----------|
| `save_context(inputs, outputs)` | Сохранить контекст в память |
| `load_memory_variables(inputs)` | Извлечь релевантную память |
| `get_system_prompt(context)` | Сгенерировать системный промпт |
| `counterfactual_query(intervention, query_nodes)` | Каузальный запрос P(Y\|do(X)) |
| `imagine_counterfactual(base_query, intervention)` | Контрфактуальное воображение |
| `get_causal_summary()` | Сводка каузальных связей |
| `get_contradictions()` | Список противоречий |
| `validate_consolidation(node_a, node_b)` | Проверка безопасности слияния |
| `create_plan(goal, tools, context)` | Планирование агента |
| `verify_hypothesis(hypothesis)` | Верификация гипотезы |
| `execute_tool(name, args)` | Выполнение инструмента |
| `evaluate_response(question, answer, contexts)` | RAGAS++ оценка |
| `compare_shadow(shadow, production)` | Shadow mode сравнение |
| `get_field_health()` | Здоровье поля |
| `trigger_healing()` | Принудительное исцеление |
| `get_cross_modal_stats()` | Статистика кросс-модальности |
| `get_meta_controller_state()` | Состояние мета-контроллера |
| `get_federated_status()` | Статус федерации |
| `get_dashboard()` | Дашборд мониторинга |
| `record_ab_metric(name, value)` | Запись A/B метрики |
| `get_stats()` | Полная статистика |
| `export_field(path)` | Экспорт в JSON |
| `import_field(path, embedder)` | Импорт из JSON |
| `clear()` | Очистка памяти |
| `rollback(n_steps)` | Откат консолидаций |
| `do_intervention(node_id, text)` | Каузальное вмешательство |
| `evolve_continuous(inputs, use_sde)` | Непрерывная эволюция ODE/SDE |
| `add_goal(description, ...)` | Добавить цель (телеология) |
| `update_goal_completion(goal_id, completion)` | Обновить прогресс цели |
| `get_active_goals()` | Получить активные цели |
| `apply_rl_feedback(response, node_ids)` | RL обратная связь |

## Команды чата (lmstudio_rtmdk_chat.py)

| Команда | Описание |
|---------|----------|
| `/stats` | Полная статистика памяти |
| `/tiers` | Распределение по уровням (episodic/semantic/procedural) |
| `/health` | Здоровье поля + топология |
| `/causal` | Каузальная сводка |
| `/contradict` | Обнаруженные противоречия |
| `/whatif {"do": {...}, "query": [...]}` | Контрфактуальный запрос P(Y\|do(X)) |
| `/imagine {"query": "...", "intervention": {...}}` | Воображение сценариев |
| `/hyperbolic` | Статистика гиперболической геометрии |
| `/predictive` | Статистика предсказательного кодирования |
| `/privacy` | Статус дифференциальной приватности |
| `/shards` | Статистика MoE-шардирования |
| `/crystallize` | Статистика кристаллизации |
| `/compression` | Статистика когнитивного сжатия |
| `/crystallize_now` | Принудительная кристаллизация |
| `/format json\|yaml\|plain` | Формат контекста |
| `/session <id>` | Переключить сессию |
| `/export` / `/clear` / `/quit` | Управление |

## Запуск с LM Studio

### CLI режим
1. Запустите LM Studio, загрузите модель, включите сервер (порт 12345)
2. Запустите: `python lmstudio_rtmdk_chat.py`

### API режим (рекомендуется)
1. Запустите сервер: `python rtmdk_server.py` или `docker-compose up -d`
2. Подключите IDE к `http://localhost:8080/v1`

## Интеграция с IDE / API

### Cursor / Continue / Aider
```json
{
  "apiBaseUrl": "http://localhost:8080/v1",
  "apiKey": "rtmdk-local"
}
```

### OpenAI SDK (Python)
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="rtmdk-local")

response = client.chat.completions.create(
    model="rtmdk",
    messages=[{"role": "user", "content": "Что я говорил вчера?"}],
    session_id="user1"
)
print(response.choices[0].message.content)
```

### cURL
```bash
# Chat с памятью
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"rtmdk","messages":[{"role":"user","content":"Привет!"}]}'

# Статистика памяти
curl http://localhost:8080/v1/memory/stats

# Здоровье поля
curl http://localhost:8080/v1/memory/health

# Контрфактуальный запрос
curl -X POST http://localhost:8080/v1/memory/imagine \
  -H "Content-Type: application/json" \
  -d '{"query":"Что если я перейду на чай?","intervention":{"n0":0.5}}'

# Do-интервенция
curl -X POST http://localhost:8080/v1/memory/intervene \
  -H "Content-Type: application/json" \
  -d '{"node_id":"n0","text":"new context"}'
```

### Docker Compose
```yaml
# docker-compose.override.yml (для подключения к LM Studio на хосте)
services:
  rtmdk-api:
    environment:
      - LM_STUDIO_URL=http://host.docker.internal:12345/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

## Тестирование

```bash
# Все тесты (326)
python -m pytest test_rtmdk_v8.py test_rtmdk_v7.py test_rtmdk_v6.py \
    test_rtmdk_v5.py test_rtmdk_v4.py test_rtmdk_v3.py \
    test_rtmdk_v2.py test_rtmdk_memory.py -v

# Только v8 (34)
python -m pytest test_rtmdk_v8.py -v

# Smoke test
python smoke_test.py
```

## Файлы проекта

| Файл | Версия | Тесты | Описание |
|------|--------|-------|----------|
| `rtmdk_memory.py` | v1 | 46 | Базовая версия |
| `rtmdk_memory_v2.py` | v2 | 47 | Адаптивные пороги, IncPCA, BM25 |
| `rtmdk_memory_v3.py` | v3 | 48 | Дифференцируемое поле, ODE, каузальность |
| `rtmdk_memory_v4.py` | v4 | 44 | Мета-адаптивность, самовосстановление |
| `rtmdk_memory_v5.py` | v5 | 23 | Каузально-топологическая память |
| `rtmdk_memory_v6.py` | v6 | 49 | Neural ODE/SDE, Agent, Production |
| `rtmdk_memory_v7.py` | v7 | 32 | Cross-modal, MetaController, Federated |
| `rtmdk_memory_v8.py` | v8 | 34 | Stratification, Hyperbolic, Predictive, Counterfactual, DP, MoE, Teleology, RL |
| `lmstudio_rtmdk_chat.py` | v8 | — | CLI чат с LM Studio |
| `rtmdk_server.py` | v8 | — | OpenAI-compatible API сервер |
| `smoke_test.py` | — | — | Быстрая проверка критических путей |
| `Dockerfile` | — | — | Docker образ |
| `docker-compose.yml` | — | — | Docker Compose конфигурация |
| `rtmdk_config.yaml` | — | — | Опциональная конфигурация |

## Критические исправления стабильности

| # | Проблема | Решение |
|---|----------|---------|
| 1 | HNSW не перехватывал query() | Авто-роутинг при N > 500 |
| 2 | Dict-мутация в consolidate() | Snapshot + deferred deletions |
| 3 | RL feedback extraction | Fallback: `1.0 - (response.count("?") + ...) * 0.15` |
| 4 | step() overload | Все операции с % N gating |
| 5 | ODE node order | Детерминированный порядок через `node_index` |
| 6 | Async workers | Lazy init в `model_post_init()` |

## Обратная совместимость

Все версии поддерживают импорт/экспорт между собой:
```python
# Экспорт из v5, импорт в v8
v5_memory.export_field("state.json")
v8_memory = RTMDKMemory.import_field("state.json", embedder)
```

## Лицензия

MIT
