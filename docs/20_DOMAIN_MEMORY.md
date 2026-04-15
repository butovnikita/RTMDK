# Phase 20: Domain Memory

> Интеграция лучших концепций из [Superagent Memory OS](https://habr.com/ru/articles/1021948/) в RTMDK

---

## Обзор

Phase 20 добавляет в RTMDK **доменную иерархию** и **управление жизненным циклом концептов**, вдохновлённые архитектурой Superagent Memory OS. При этом сохраняются все преимущества RTMDK: гиперболическая геометрия, ODE/SDE динамика, do-calculus, диалектическая консолидация.

### Что добавлено

| Концепция | Источник | Реализация в RTMDK |
|-----------|----------|-------------------|
| **Domain Hierarchy** | `Domain→AnchorEntity→Topic→Concept` | `domain`, `subdomain`, `topic` поля в узле |
| **ConceptHypothesis Lifecycle** | `create/attach/refine/reject` | `state`, `confidence`, `revision_count`, `conflict_with` |
| **Evidence Spans** | Для legal/medical traceability | `evidence_spans: List[Dict]` |
| **Bi-temporal Facts** | ADR-019: `valid_time` + `system_time` | `valid_from`, `valid_until`, `fact_state`, `superseded_by` |
| **Domain-aware Retrieval** | Hierarchical top-down navigation | Фильтрация по домену при retrieval |
| **Cross-domain Guard** | Concept Memory Loop | Запрет консолидации узлов из разных доменов |

---

## Новые поля MemoryNode

### Track 1: Domain Hierarchy

```python
domain: str = "general"        # Макро-домен: "IT", "Law", "Medicine", "Finance", "Science"
subdomain: str = ""            # Поддомен: "Databases", "Contracts", "Cardiology"
topic: str = ""                # Топик/AnchorEntity: "SQL", "Employment Law"
```

**Автоматическое определение:** `detect_domain(text)` — pattern-based классификатор (~0.1ms, кэшируется).

### Track 2: Concept Lifecycle

```python
state: str = "stable"          # stable / weakened / disputed / broken / stale / archived
confidence: float = 1.0        # Уверенность в концепте (0.0–1.0)
revision_count: int = 0        # Сколько раз пересматривался
conflict_with: List[str] = []  # ID узлов, с которыми конфликтует
```

### Track 3: Evidence Spans

```python
evidence_spans: List[Dict] = []
# Каждый: {"source_id": str, "text": str, "confidence": float}
```

Для юридической и медицинской памяти — полная прослеживаемость источников.

### Track 4: Bi-temporal Facts

```python
valid_from: Optional[float] = None    # Когда факт стал верен (unix timestamp)
valid_until: Optional[float] = None   # Когда факт устарел
fact_state: str = "active"            # active / stale / disputed / rejected / archived
superseded_by: Optional[str] = None   # ID узла-заменителя
```

---

## Конфигурация

### Config Flags

```python
domain_aware_retrieval: bool = False    # OFF по умолчанию — zero regression risk
domain_consolidation_guard: bool = True # ON по умолчанию — защита от cross-domain merge
```

### Environment Variables

```bash
RTMDK_DOMAIN_AWARE_RETRIEVAL=true/false
RTMDK_DOMAIN_CONSOLIDATION_GUARD=true/false
```

### Presets

| Preset | domain_aware_retrieval | domain_consolidation_guard |
|--------|:----------------------:|:--------------------------:|
| `local()` | ❌ | ✅ |
| `production()` | ❌ | ✅ |
| `legal()` | ✅ | ✅ |
| `medical()` | ✅ | ✅ |
| `research()` | ❌ | ✅ |

---

## Как работает

### 1. Сохранение контекста (`save_context()`)

```
text → detect_domain() → (domain, subdomain, topic)
      → detect_tier()  → (episodic/semantic/procedural)
      → detect_modality() → (text/code/audio/vision/metrics)
      → create node with all metadata
```

### 2. Retrieval (`load_memory_variables()`)

```
Query → detect_domain(query)
      → if domain_aware_retrieval:
          filter results by domain
          fallback to global if < top_k/2
      → format context with [DOM:XXX] tokens
```

### 3. Консолидация (`consolidate()`)

```
For each high-tension pair (node1, node2):
  → if domain_consolidation_guard:
      if node1.domain != node2.domain AND both != "general":
          SKIP consolidation
          mark as conflict_with
  → if bi-temporal check:
      if node1.fact_state != "active" OR node2.fact_state != "active":
          SKIP consolidation
  → proceed with normal consolidation
```

---

## Формат контекста (ATTENTION)

```
### ATTENTION_CONTEXT
[ATTN:0.842][SAL:0.601][TIER:S][DOM:IT][CAUSAL:2] U:How to index SQL? | AI:Use CREATE INDEX
[ATTN:0.756][SAL:0.523][TIER:E][DOM:Law][STATE:D] U:Contract clause? | AI:See section 5.2
```

Новые токены:
- `[DOM:XXX]` — домен (только если != "general")
- `[STATE:X]` — состояние концепта (только если != "stable")
- `[SRC:XXXXXXXX]` — источник evidence (для legal/medical)

---

## Поддерживаемые домены

| Домен | Поддомены |
|-------|-----------|
| **IT** | Databases, Programming, DevOps, Security, Networking |
| **Law** | Contracts, Employment, Intellectual Property, Litigation |
| **Medicine** | Cardiology, Pharmacology, Surgery, Diagnostics |
| **Finance** | Banking, Investing, Accounting, Insurance |
| **Science** | Physics, Chemistry, Biology, Mathematics |
| **Education** | Teaching, Research |
| **Business** | Management, Marketing, Sales |
| **general** | (fallback) |

---

## Backward Compatibility

✅ **100% backward compatible:**
- Все новые поля имеют `default` значения
- Старые `memory.json` файлы загружаются без ошибок
- `domain_aware_retrieval` OFF по умолчанию → поведение идентично baseline
- Нет breaking changes в API

---

## Тестирование

```bash
# Запустить все тесты Phase 20
pytest tests/test_domain_memory.py -v

# Тесты покрывают:
# 1. Сериализация/десериализация новых полей
# 2. Загрузка старых memory.json без ошибок
# 3. detect_domain() корректность
# 4. Domain-aware retrieval фильтрация
# 5. Cross-domain consolidation guard
# 6. Bi-temporal facts
# 7. Concept state transitions
# 8. Evidence spans
# 9. Legal/medical presets
# 10. ATTENTION контекст токены
```

---

## Источники

- [Superagent Memory OS (Habr)](https://habr.com/ru/articles/1021948/)
- Концепции: ConceptHypothesis, Data Contract, Concept Memory Loop, Bi-temporal facts, Agentic Gardener
- Адаптировано для RTMDK резонансно-топологической архитектуры
