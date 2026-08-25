# RTMDK Calibration Reference & Values Guide

> ⚠️ **R3.3 (2026-08-24, audit/risks-2026-08-24): частично устарел.** Полный источник правды — `rtmdk/memory/config.py` (230+ полей, 9 групп, 59 env-override, `ORPHANED_FLAGS` 36) и `rtmdk/config.py` (пресеты). Этот файл покрывает только базовые 30 параметров; новые `pipeline_breaker_*`, `sot_*`, `tiered_*`, `conformal_*`, `kalman_*` см. в коде и `docs/05_FINE_TUNING.md`. Планируется генерация `Values.md` из `RTMDKConfig` (BACKLOG.md R3.3, docs/RISKS.md).

Полная документация по калибровке переменных, гиперпараметров и конфигураций системы RTMDK.
Данные актуальны для `rtmdk/memory/config.py` (канонично) и `rtmdk/support/swarm.py`.

---

## 1. Основные параметры памяти (Memory Physics)
Влияют на ёмкость, время жизни узлов и точность поиска.

| Параметр | Значение (Default) | Описание | Рекомендации | 
|:---|:---|:---|:---|
| **`rtmdk.latent_dim`** | `64` | Размерность скрытого многообразия (вектор узла). | Повышение до `128` улучшает детализацию, но требует больше RAM/CPU. |
| **`rtmdk.embedding_dim`** | `768` | Размерность выходного вектора эмбеддеров (nomic-embed-text). | Должно соответствовать выбранной модели. |
| **`rtmdk.decay_rate`** | `0.997` | Коэффициент затухания значимости (salience) за шаг. | `< 0.95` для быстрой смены контекста. `> 0.999` для долгосрочной памяти. |
| **`rtmdk.top_k`** | `5` | Количество узлов, отдаваемых в контекст LLM. | Увеличить до `10-15` для сложных запросов (риск переполнения контекста). |
| **`rtmdk.tension_threshold`** | `0.15` | Порог "напряжения" поля, запускающий консолидацию. | Снижение (`0.1`) = частая оптимизация (CPU load). Повышение (`0.3`) = реже. |

---

## 2. Мета-когнитивные параметры (Introspective Meta-Memory)
Настройки для `MetaMemoryEvaluator`, `RLFeedbackLoop` и `PredictiveCoder`.

| Параметр | Значение (Default) | Описание | Рекомендации | 
|:---|:---|:---|:---|
| **`self_sup_threshold`** | `0.3` | Минимальная доверительная оценка (confidence) для самообучения. | Повышение (`0.5`) снижает ложные обучения, но замедляет адаптацию. |
| **`self_reflection_freq`** | `100` | Интервал шагов (итераций диалога) для запуска мета-анализа. | Для коротких сессий уменьшить до `20-50`. |
| **`self_sup_verify_after_consolidate`** | `False` | Валидировать качество памяти после каждой консолидации. | Включить (`True`) для стабильности в production (накладные расходы CPU). |

---

## 3. Безопасность и Приватность (Security & DP)
Настройки `DifferentialPrivacy` и защиты от инъекций.

| Параметр | Значение (Default) | Описание | Рекомендации | 
|:---|:---|:---|:---|
| **`dp.epsilon`** | `2.0` | Бюджет приватности. Обратная величина шуму. | `1.0` = строгая анонимность (сильный шум). `10.0` = минимум шума (высокая точность). |
| **`dp.delta`** | `0.00001` | Вероятность выхода за пределы бюджета. | Обычно фиксируется на `1e-5`. |
| **`dp.max_norm`** | `1.0` | Предел обрезки (clipping) для обновления весов/сумм. | Защита от аномальных выбросов (outliers). Снижать при "всплесках" памяти. |

---

## 4. Роевая память (Swarm Memory)
Параметры `swarm_memory.py` и `SwarmConsensusProtocol`.

| Параметр | Значение (Default) | Описание | Рекомендации | 
|:---|:---|:---|:---|
| **`max_scenarios`** | `5` | Кол-во контрфактуальных сценариев для симуляции. | Влияет на качество предсказаний. Увеличить до `10` при наличии ресурсов. |
| **`KuramotoSync.Coupling`** | *(auto)* | Сила связи между узлами (зависит от топологии). | Определяет, насколько быстро синхронизируются кластеры памяти. |

---

## 5. Управление Фичами (Phase Flags)
Переключатели в `rtmdk_config.yaml`. Включают/выключают сложные модули.

```yaml
rtmdk:
  causal_topological: true       # Вкл/Выкл каузальный анализ связей
  do_calculus_validation: true   # Фильтрация паразитных связей (Do-calculus)
  meta_adaptive: true            # Самокалибровка параметров (Meta-Loop)
  self_healing: true             # Автоматическое восстановление целостности графа
  predictive_coding: true        # Предсказательное кодирование (экономия ресурсов)
  hyperbolic: false              # Гиперболическая геометрия (экспериментально)
  differential_privacy: false    # Добавить дифференциальный шум
  counterfactual_imagination: true # Генерация гипотетических узлов
```

## 6. Сервер и Интеграция (Infrastructure)

| Параметр | Значение (Default) | Описание |
|:---|:---|:---|
| **`server.host`** | `0.0.0.0` | Адрес слушателя API. |
| **`server.port`** | `8080` | Порт API (совместим с OpenAI). |
| **`lm_studio.url`** | `localhost:12345` | Адрес локального LLM-инференса. |
| **`memory.auto_save_interval`**| `60` | Интервал сохранения (сек). |

---

## 7. Pipeline & SOT (добавлено в R3.3 — ранее отсутствовало, см. `rtmdk/memory/config.py:748`)

| Параметр | Значение (Default) | Описание | Где |
|:---|:---|:---|:---|
| **`pipeline_breaker_thresholds`** | `{"embed":5000,"route":100,"retrieve":500,"rerank":1000,"calibrate":200,"explain":100}` мс | Per-stage SLO; `validate()` помечает ≤0 как `ERROR:` (R3.1) | `ProductionConfig` |
| **`pipeline_breaker_failure_threshold`** | `5` | Трип при ≥N фейлов | `ProductionConfig` |
| **`pipeline_breaker_recovery_timeout_ms`** | `30000` | Восстановление half-open | `ProductionConfig` |
| **`sot_enabled`** | `False` | Вкл. Self-Organizing Tokenizer | `SOTConfig` |
| **`sot_max_vocab`** | `4096` | Max vocab; sparse PMI >5000 (`sif_embedder.py:168`) | `SOTConfig` |
| **`sot_skipgram_window`** | `1` | 1=adjacent, >1 OOM риск (R6.2) | `SOTConfig` |
| **`sot_max_cooccurrence`** | `100000` | Лимит COOC до pruning | `SOTConfig` |
| **`tiered_storage_enabled`** | `False` | Tiered v1 | `MemorySystemConfig` |
| **`tiered_storage_v2_enabled`** | `False` | Tiered v2 (memmap/LFU) | `MemorySystemConfig` |
| **`conformal_prediction`** | `False` | ICP; нужен calibrate (`conformal_min_calib=50`) | `CoreConfig` |

> Источник: `rtmdk/memory/config.py` — `git grep -n "pipeline_breaker\|sot_" rtmdk/memory/config.py` (см. R3.3).
