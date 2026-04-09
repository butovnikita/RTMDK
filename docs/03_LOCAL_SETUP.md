# RTMDK — Инструкция по локальному использованию через Docker Compose

## Быстрый старт за 5 минут

### 1. Запуск

```bash
cd C:\Users\Никита\Desktop\llm_lab
docker-compose up -d
```

Сервер запущен на `http://localhost:8080`

### 2. Проверка

```bash
curl http://localhost:8080/health
# → {"status": "ok", "version": "8.0.0", "lm_studio": false, "memory_nodes": 0}
```

### 3. Использование через Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="rtmdk-local"
)

response = client.chat.completions.create(
    model="rtmdk",
    messages=[{"role": "user", "content": "Привет, это мой тестовый запрос"}],
    session_id="local-user"
)
print(response.choices[0].message.content)
```

---

## Полная инструкция по локальному использованию

### Предварительные требования

| Компонент | Зачем | Обязательно? |
|-----------|-------|-------------|
| Docker Desktop | Запуск RTMDK сервера | ✅ Да |
| LM Studio | Реальные LLM + эмбеддинги | ❌ Опционально |
| Python 3.10+ | Клиентские скрипты | ❌ Опционально |

### Вариант A: Только RTMDK сервер (без LLM)

```bash
# Запуск
docker-compose up -d

# Проверить статус
docker-compose ps

# Посмотреть логи
docker-compose logs -f rtmdk-memory
```

**Что работает:**
- ✅ Сохранение/запрос памяти через API
- ✅ Каузальные запросы
- ✅ Контрфактуальное воображение
- ✅ Статистика и здоровье поля
- ❌ Чат с LLM (нужен LM Studio)

**Что НЕ работает:**
- ❌ `/v1/chat/completions` — вернёт 503 (нет LLM)
- ❌ Эмбеддинги через LM Studio — fallback на случайные векторы

### Вариант B: RTMDK + LM Studio (полный функционал)

1. **Запусти LM Studio:**
   - Открой LM Studio
   - Загрузи модель (например, Qwen2.5-7B)
   - Включи сервер: Server → Start на порту `12345`
   - Загрузи модель эмбеддингов: `nomic-embed-text-v1.5`

2. **Обнови docker-compose для доступа к хосту:**

Создай файл `docker-compose.override.yml`:
```yaml
services:
  rtmdk-api:
    environment:
      - RTMDK_ENABLE_LM_STUDIO=true
      - LM_STUDIO_URL=http://host.docker.internal:12345/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

3. **Перезапусти:**
```bash
docker-compose down
docker-compose up -d
```

4. **Проверь интеграцию:**
```bash
curl http://localhost:8080/health
# → "lm_studio": true — значит подключено!
```

### Вариант C: Локальный Python (без Docker)

Если не хочешь использовать Docker:

```bash
cd C:\Users\Никита\Desktop\llm_lab

# Установи зависимости
pip install fastapi uvicorn numpy scipy pydantic requests

# Запусти сервер
python rtmdk_server.py

# Сервер на http://localhost:8080
```

---

## Использование CLI-чата

```bash
# С LM Studio
python lmstudio_rtmdk_chat.py

# Без LM Studio (только память)
# Запусти сначала сервер:
python rtmdk_server.py
```

### Команды чата

```
/stats          → полная статистика
/tiers          → распределение по уровням памяти
/health         → здоровье поля
/causal         → каузальная сводка
/contradict     → противоречия
/whatif {...}   → контрфактуальный запрос
/imagine {...}  → воображение сценариев
/hyperbolic     → гиперболическая геометрия
/predictive     → предсказательное кодирование
/privacy        → дифференциальная приватность
/shards         → MoE-шардирование
/crystallize    → кристаллизация
/compression    → когнитивное сжатие
/format json    → формат контекста
/session user1  → переключить сессию
/export         → экспорт памяти
/clear          → очистить память
/quit           → выйти
```

---

## Использование Streamlit Dashboard

```bash
pip install streamlit matplotlib pandas
streamlit run streamlit_app.py
# → http://localhost:8501
```

**Вкладки:**
- 💬 Chat — чат с памятью
- 🗺️ Field — визуализация поля (2D проекция)
- 🎯 Goals — управление целями
- 🔒 Security — монитор безопасности
- 📦 Nodes — таблица всех узлов

---

## Тестирование

```bash
# Быстрая проверка
python smoke_test.py

# Eval pipeline
python eval_pipeline.py --n_samples 50

# Swarm симуляция
python swarm_memory.py --n_agents 5 --n_rounds 10
```

---

## Управление Docker-контейнером

```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Остановка с удалением данных
docker-compose down -v

# Пересборка
docker-compose up -d --build

# Логи
docker-compose logs -f rtmdk-memory

# Войти в контейнер
docker exec -it rtmdk-memory sh

# Проверить здоровье
docker-compose ps
curl http://localhost:8080/health
```

---

## Файлы памяти

| Файл | Описание |
|------|----------|
| `~/.rtmdk/memory.json` | Автосохранение (каждые 60 сек) |
| `rtmdk_lmstudio_state.json` | Состояние CLI-сессии |
| `eval_report.json` | Результаты eval pipeline |
| `swarm_report.json` | Отчёт роевой памяти |

---

## Конфигурация через переменные окружения

```yaml
# docker-compose.yml
environment:
  - RTMDK_HOST=0.0.0.0              # Хост сервера
  - RTMDK_PORT=8080                  # Порт
  - RTMDK_MEMORY_FILE=/data/memory.json  # Файл памяти
  - RTMDK_ENABLE_LM_STUDIO=false     # Включить LM Studio
  - RTMDK_AUTO_SAVE=60               # Интервал автосохранения (сек)
  - RTMDK_API_KEY=rtmdk-local        # API ключ
```

---

## Примеры использования API

### Сохранить память

```bash
curl -X POST http://localhost:8080/v1/memory/save \
  -H "Content-Type: application/json" \
  -d '{"input":"Меня зовут Никита","output":"Запомнил: Никита"}'
```

### Запросить память

```bash
curl -X POST http://localhost:8080/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query":"как меня зовут?","session_id":"default"}'
```

### Контрфактуальный запрос

```bash
curl -X POST http://localhost:8080/v1/memory/imagine \
  -H "Content-Type: application/json" \
  -d '{"query":"Что если я перейду на чай?","intervention":{"n0":0.5}}'
```

### Статистика

```bash
curl http://localhost:8080/v1/memory/stats
curl http://localhost:8080/v1/memory/health
```

---

## Отладка и улучшение

### Как отлаживать

1. **Логи контейнера:**
   ```bash
   docker-compose logs -f rtmdk-memory
   ```

2. **Проверка здоровья:**
   ```bash
   curl http://localhost:8080/v1/memory/health
   ```

3. **Smoke test:**
   ```bash
   python smoke_test.py
   ```

4. **Python REPL:**
   ```python
   from rtmdk import RTMDKMemory, RTMDKConfig
   # Импортируй и тестируй напрямую
   ```

### Как улучшать

1. **Добавить новые фичи:**
   - Модифицируй `rtmdk_memory_v8.py` или модули в `rtmdk/`
   - Пересобери: `docker-compose up -d --build`

2. **Изменить конфигурацию:**
   - Правь `docker-compose.yml` или создай `.env`
   - Перезапусти: `docker-compose restart`

3. **Добавить тесты:**
   - Новый файл `test_my_feature.py`
   - Запусти: `python -m pytest test_my_feature.py -v`

4. **Профилирование:**
   ```python
   import cProfile
   cProfile.run('memory.save_context(...)', 'profile.stats')
   # → python -m pstats profile.stats
   ```

### Частые проблемы

| Проблема | Решение |
|----------|---------|
| Port 8080 занят | Смени порт в `docker-compose.yml`: `"8081:8080"` |
| LM Studio не подключается | Проверь `host.docker.internal` в override |
| Память не сохраняется | Проверь права на `~/.rtmdk/` |
| Docker не запускается | `docker-compose down -v && docker-compose up -d --build` |

---

*Инструкция актуальна для коммита `98c49d0` — Phase 15.*
