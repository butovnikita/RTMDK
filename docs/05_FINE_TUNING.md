# RTMDK Fine-Tuning Guide

> Полное руководство по настройке RTMDK для максимальной эффективности.
> Версия: 3.0 (unified config, env vars, 8 профилей)

---

## Оглавление

1. [Быстрый старт](#1-быстрый-старт)
2. [Env Var Overrides](#2-env-var-overrides) — **НОВОЕ**
3. [Runtime Config через API](#3-runtime-config-через-api) — **НОВОЕ**
4. [Core переменные](#4-core-переменные)
5. [Retrieval переменные](#5-retrieval-переменные)
6. [Performance переменные](#6-performance-переменные)
7. [Production переменные](#7-production-переменные)
8. [Phase 18: Энграммы](#8-phase-18-энграммы)
9. [Phase 19: Advanced](#9-phase-19-advanced)
10. [Scaling переменные](#10-scaling-переменные)
11. [8 Профилей](#11-8-профилей)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Быстрый старт

```python
from rtmdk import RTMDKConfig

# Выбери пресет
config = RTMDKConfig.local()       # Персональный ассистент
config = RTMDKConfig.production()  # Продакшен сервер
config = RTMDKConfig.research()    # Максимальная точность
config = RTMDKConfig.enterprise()  # 100K+ узлов

# Или настрой вручную
config = RTMDKConfig(
    latent_dim=64,
    top_k=5,
    decay_rate=0.997,
)
```

---

## 2. Env Var Overrides

Любой из 59 параметров можно переопределить через `RTMDK_*` env var.
Приоритет: **явные аргументы > env vars > defaults пресета**.

```bash
# Выбрать пресет
RTMDK_PRESET=production python rtmdk_server.py

# Переопределить параметры пресета
RTMDK_PRESET=local RTMDK_LATENT_DIM=128 RTMDK_TOP_K=10 python rtmdk_server.py

# Полный контроль
RTMDK_PRESET=research \
  RTMDK_DECAY_RATE=0.9995 \
  RTMDK_CAUSAL_TOPOLOGICAL=true \
  python rtmdk_server.py
```

**Все поддерживаемые env vars:**

| Категория | Переменные |
|-----------|-----------|
| **Core** | `RTMDK_EMBEDDING_DIM`, `RTMDK_LATENT_DIM`, `RTMDK_DECAY_RATE`, `RTMDK_TENSION_THRESHOLD`, `RTMDK_MIN_RESPONSE`, `RTMDK_TOP_K`, `RTMDK_MAX_NODES`, `RTMDK_CONSOLIDATION_MODE` |
| **Retrieval** | `RTMDK_PHASE_COUPLING`, `RTMDK_BANDWIDTH`, `RTMDK_USE_HNSW`, `RTMDK_HNSW_M`, `RTMDK_BM25_FALLBACK`, `RTMDK_LEARN_PROJECTION`, `RTMDK_PROJECTION_LR`, `RTMDK_PROJECTION_UPDATE_FREQ` |
| **Performance** | `RTMDK_ENABLE_ASYNC`, `RTMDK_SOFT_GATES`, `RTMDK_ATTENTION_BIAS`, `RTMDK_ADAPTIVE_THRESHOLD` |
| **Production** | `RTMDK_CROSS_MODAL`, `RTMDK_CAUSAL_TOPOLOGICAL`, `RTMDK_META_ADAPTIVE`, `RTMDK_SELF_HEALING`, `RTMDK_VERSION_CONTROL` |
| **Engrams** | `RTMDK_ENABLE_ENGRAMS`, `RTMDK_ENGRAM_MIN_NODES`, `RTMDK_ENGRAM_MAX_NODES` |
| **Phase 19** | `RTMDK_OFFLINE_DREAMING`, `RTMDK_CAUSAL_TRAVERSAL`, `RTMDK_CAUSAL_MAX_HOPS`, `RTMDK_SSM_DYNAMICS`, `RTMDK_SSM_STATE_DIM`, `RTMDK_TRUST_CONSENSUS`, `RTMDK_NEURO_SYMBOLIC_PROVER` |
| **Phase 11** | `RTMDK_HYPERBOLIC`, `RTMDK_PREDICTIVE_CODING`, `RTMDK_COUNTERFACTUAL_IMAGINATION`, `RTMDK_DIFFERENTIAL_PRIVACY`, `RTMDK_DP_EPSILON` |
| **Phase 12-17** | `RTMDK_SPARSE_ROUTING`, `RTMDK_NUM_SHARDS`, `RTMDK_GOAL_TRACKING`, `RTMDK_RL_FEEDBACK`, `RTMDK_LOW_RANK_COMPRESSION`, `RTMDK_META_MEMORY`, `RTMDK_SECURITY_ENABLED`, `RTMDK_SWARM_MEMORY`, `RTMDK_SYMBOLIC_OVERLAY`, `RTMDK_SAFETY_CERTIFIER`, `RTMDK_ROLE_SHARDING` |
| **Misc** | `RTMDK_CONTEXT_FORMAT`, `RTMDK_LOG_LEVEL` |

---

## 3. Runtime Config через API

```bash
# Изменить пресет (сохраняется в .env, требует перезапуска)
curl -X POST http://localhost:8080/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{"RTMDK_PRESET": "production"}'
# Ответ: {"status":"ok", "needs_restart": true, "restart_required_keys": ["RTMDK_PRESET"]}

# Изменить гиперпараметры
curl -X POST http://localhost:8080/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{"RTMDK_LATENT_DIM": "128", "RTMDK_TOP_K": "10"}'

# Изменить серверные настройки (применяются сразу)
curl -X POST http://localhost:8080/v1/admin/config \
  -H "Content-Type: application/json" \
  -d '{"RTMDK_EMBED_MODEL": "text-embedding-3-small"}'
```

---

## 4. Core переменные

### embedding_dim
**Что:** Размерность входных эмбеддингов
**По умолчанию:** 768
**Диапазон:** 384–1024
**Влияние:**
- ↑ 1024 = больше семантической информации, но +33% RAM
- ↓ 384 = быстрее, но может терять нюансы
**Когда менять:** Только если меняешь модель эмбеддера

### latent_dim
**Что:** Внутренний размер представления памяти
**По умолчанию:** 256
**Диапазон:** 64–512
**Влияние:**
- ↑ 512 = Recall@1 +5-10%, RAM +2x, latency +30%
- ↓ 64 = Recall@1 -15-20%, RAM /4, latency -40%
**Рекомендация:** 256 для баланса, 512 для research

### decay_rate
**Что:** Скорость забывания
**По умолчанию:** 0.999
**Диапазон:** 0.980–0.9999
**Влияние:**
- 0.999 → half-life 693 шага (очень медленно)
- 0.995 → half-life 138 шагов (умеренно)
- 0.990 → half-life 69 шагов (быстро)
- 0.980 → half-life 34 шага (очень быстро)
**Когда менять:**
- ↓ 0.995 если память "засоряется" нерелевантным
- ↑ 0.9995 если важное забывается слишком быстро

### min_response
**Что:** Минимальный порог резонанса
**По умолчанию:** 0.005
**Диапазон:** 0.001–0.1
**Влияние:**
- ↓ 0.001 = больше результатов (включая слабые), выше recall но больше шума
- ↑ 0.05 = только сильные совпадения, ниже recall но выше precision
**Рекомендация:** 0.005 для общего использования

### top_k
**Что:** Сколько узлов возвращать
**По умолчанию:** 5
**Диапазон:** 1–20
**Влияние:**
- ↑ 10 = больше контекста для LLM (+recall), но +2x токенов
- ↓ 3 = экономия токенов, но может пропустить важное
**Рекомендация:** 5 для баланса, 10 для сложных запросов

### tension_threshold
**Что:** Порог консолидации (слияния узлов)
**По умолчанию:** 0.25
**Диапазон:** 0.10–0.50
**Влияние:**
- ↓ 0.15 = частая консолидация, экономия RAM, риск потери деталей
- ↑ 0.40 = редкая консолидация, больше деталей, больше RAM
**Рекомендация:** 0.25 для общего случая

---

## 3. Retrieval переменные

### phase_coupling
**Что:** Влияние фазового выравнивания на резонанс
**По умолчанию:** 0.3
**Диапазон:** 0.0–1.0
**Влияние:**
- 0.0 = фаза игнорируется (только spatial)
- 0.3 = умеренное влияние (default)
- 1.0 = фаза доминирует
**Рекомендация:** 0.3 — проверено empirically

### bandwidth
**Что:** Ширина ядра резонанса
**По умолчанию:** 1.0
**Диапазон:** 0.3–3.0
**Влияние:**
- ↓ 0.5 = узкий фокус, точные совпадения
- ↑ 2.0 = широкий фокус, общие совпадения
**Когда менять:** ↓ для специализированных баз, ↑ для общих

### use_hnsw
**Что:** Включить HNSW поиск
**По умолчанию:** True
**Влияние:**
- True = O(log N) поиск — критично для N > 100
- False = O(N) brute force — только для отладки
**Рекомендация:** Всегда True

### hnsw_m
**Что:** Связей на узел в HNSW
**По умолчанию:** 16
**Диапазон:** 8–64
**Влияние:**
- 8 = быстрый build, ниже recall
- 16 = баланс (default)
- 32 = лучший recall для N > 10K
- 64 = максимальный recall, +50% RAM на индекс
**Рекомендация:** 16 для N < 10K, 32 для N > 10K

### bm25_fallback
**Что:** Использовать BM25 при низком резонансе
**По умолчанию:** True
**Влияние:**
- True = 0% "мёртвых" запросов
- False = может вернуть пустой контекст
**Рекомендация:** Всегда True для продакшена

---

## 4. Performance переменные

### enable_async
**Что:** Асинхронная обработка
**По умолчанию:** False
**Влияние:**
- True = non-blocking saves, выше throughput
- False = синхронно, проще для отладки
**Когда менять:** True для серверов с высоким QPS

### attention_bias
**Что:** Структурное внимание к результатам
**По умолчанию:** True
**Влияние:**
- True = буст каузально-связанных узлов (+5-10% relevance)
- False = raw scores
**Рекомендация:** Всегда True

---

## 5. Production переменные

### version_control
**Что:** Дельта-версионирование памяти
**По умолчанию:** False
**Влияние:**
- True = можно откатить к любой версии, +10% RAM
- False = нет истории, экономия RAM
**Когда менять:** True для продакшена с отладкой

### causal_topological
**Что:** Обнаружение каузальных связей
**По умолчанию:** False
**Влияние:**
- True = находит причинно-следственные связи, O(N²) overhead
- False = без каузального графа
**Когда менять:** Только для causal analysis, не для обычного retrieval

---

## 7. Phase 18: Энграммы

### enable_engrams
**Что:** Группировка коактивированных узлов в единые воспоминания
**По умолчанию:** True
**Влияние:**
- ↑ True = pattern completion, 3x faster search
- ↓ False = стандартный поиск по узлам

### engram_min_nodes / engram_max_nodes
**Что:** Мин/макс узлов в одной энграмме
**По умолчанию:** 2 / 20
**Когда менять:** ↓ min=5 для только сильных воспоминаний

### engram_creation_threshold
**Что:** Порог создания энграммы [0-1]
**По умолчанию:** 0.6
**Влияние:** ↓ 0.4 = лёгкое создание, ↑ 0.8 = только сильные

### engram_pattern_completion
**Что:** Дополнение частичного запроса до полной энграммы
**По умолчанию:** True
**Влияние:** 20% совпадение → 100% воспоминание

---

## 8. Phase 19: Advanced

### offline_dreaming
**Что:** Фоновые циклы для TDA, кристаллизации, topology repair
**По умолчанию:** True (кроме local/streaming)
**Влияние:**
- ↑ True = -90% latency spikes
- ↓ False = всё в real-time (может тормозить)

### causal_traversal
**Что:** Поиск по каузальному графу при retrieval
**По умолчанию:** True
**Влияние:** +15-25% на "почему"-вопросах, +5ms к latency

### ssm_dynamics
**Что:** State Space Models (O(N)) вместо NeuralODE (O(N³))
**По умолчанию:** True для production/enterprise/streaming
**Когда менять:** True для N > 10K

### trust_consensus
**Что:** DAG доверия для федерации
**По умолчанию:** False
**Когда менять:** True для multi-agent deployments

### neuro_symbolic_prover
**Что:** Z3/Prolog для разрешения противоречий
**По умолчанию:** False
**Когда менять:** True для legal/medical доменов

---

## 9. Scaling переменные

### sparse_routing
**Что:** MoE-style шардирование
**По умолчанию:** False
**Влияние:**
- True = маршрутизация к релевантным шардам
- False = глобальный поиск
**Когда менять:** Только для N > 50K

### num_shards
**Что:** Количество шардов
**По умолчанию:** 8
**Формула:** `sqrt(N / 1000)`
**Примеры:**
- 10K узлов → ~3 шарда
- 100K узлов → ~10 шардов
- 1M узлов → ~32 шарда *(прогноз; распределённое шардирование не реализовано в v8.3)*

---

## 10. 8 Профилей

### RTMDKConfig.local()
```python
# Персональный ассистент
# RAM: ~16MB, Latency: ~5ms, Nodes: до 10K
latent_dim=256, top_k=5, decay_rate=0.999
enable_engrams=True, offline_dreaming=False
```

### RTMDKConfig.production()
```python
# Продакшен сервер
# RAM: ~50MB, Latency: ~6ms, Nodes: до 100K
latent_dim=256, top_k=5, ssm_dynamics=True
offline_dreaming=True, trust_consensus=True
```

### RTMDKConfig.research()
```python
# Максимальная точность
# RAM: ~200MB, Latency: ~50ms, Nodes: unlimited
latent_dim=512, top_k=10, neuro_symbolic_prover=True
causal_max_hops=5
```

### RTMDKConfig.enterprise()
```python
# Распределённая система (roadmap, не реализовано в v8.3)
# RAM: ~250MB/shard, Latency: ~15ms, Nodes: 500K+
ssm_state_dim=128, sparse_routing=True, num_shards=32
```

### RTMDKConfig.agent()
```python
# Автономный агент
# С активным выводом и каузальным поиском
causal_max_hops=4, ssm_dynamics=True
```

### RTMDKConfig.legal()
```python
# Юриспруденция
# Z3 prover для обнаружения противоречий
neuro_symbolic_prover=True, prover_backend="z3"
causal_max_hops=5
```

### RTMDKConfig.medical()
```python
# Медицина
# Высокий trust + audit trail
trust_min_reputation=0.5, neuro_symbolic_prover=True
version_control=True
```

### RTMDKConfig.streaming()
```python
# High-throughput real-time
# RAM: ~30MB, Latency: ~3ms
offline_dreaming=False, causal_traversal=False
ssm_dynamics=True, attention_bias=False
```

---

## 11. Troubleshooting

| Симптом | Причина | Решение |
|---------|---------|---------|
| **Слишком много RAM** | latent_dim высокий или max_nodes=None | ↓ latent_dim до 128, ↑ max_nodes лимит |
| **Низкий recall** | min_response слишком высокий | ↓ min_response до 0.001, ↑ top_k до 10 |
| **Медленные запросы** | HNSW отключён или N слишком большой | ↑ use_hnsw=True, ↑ hnsw_ef_construction |
| **Память "засоряется"** | decay_rate слишком высокий | ↓ decay_rate до 0.995 |
| **Важное забывается** | decay_rate слишком низкий | ↑ decay_rate до 0.9995 |
| **Пустые ответы** | bm25_fallback=False | ↑ bm25_fallback=True |
| **Консолидация ломает данные** | tension_threshold слишком низкий | ↑ tension_threshold до 0.35 |
| **Нет каузальных связей** | causal_topological=False | ↑ causal_topological=True (медленно) |

---

*Последнее обновление: Май 2026, RTMDK v8.3*
