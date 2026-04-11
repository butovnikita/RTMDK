# RTMDK — Resonance-Topological Memory v8.0

> Долгосрочная память для LLM на основе резонансной топологии и диалектической консолидации

---

## 📦 Две версии проекта

Проект разделён на **две конфигурации** для разных сценариев использования:

| | 🏭 Production | 🏠 Home / SillyTavern |
|---|---|---|
| **Назначение** | API-сервер для продакшена, микросервисы | Домашнее использование с SillyTavern |
| **Порты** | `8080` (один) | `8080` + `5000` (proxy) |
| **SillyTavern** | ❌ Не включён | ✅ Полная поддержка |
| **Dashboard** | ✅ Веб-UI | ✅ Веб-UI |
| **Dockerfile** | `Dockerfile` | `Dockerfile.home` |
| **Compose** | `docker-compose.prod.yml` | `docker-compose.home.yml` |
| **Зависимости** | Минимальные | Полные |
| **Запуск** | `start_production.bat` / `start_production.py` | `start_sillytavern.bat` / `rtmdk_server.py` |

---

## 🚀 Быстрый старт

### 🏭 Production (без SillyTavern)

```bash
# 1. Установить минимальные зависимости
pip install -r requirements-prod.txt

# 2. Запустить сервер
python start_production.py

# Или через Windows batch-файл:
start_production.bat

# Docker
docker-compose -f docker-compose.prod.yml up -d
```

### 🏠 Home / SillyTavern

```bash
# 1. Установить полные зависимости
pip install -r requirements-home.txt

# 2. Запустить сервер + SillyTavern proxy
python rtmdk_sillytavern_launcher.py

# Или через Windows batch-файл:
start_sillytavern.bat

# Docker
docker-compose -f docker-compose.home.yml up -d
```

### 🔧 Monolith (один процесс, ST endpoints встроены)

```bash
# Альтернативный вариант — всё в одном процессе
python rtmdk_server.py
```

> Monolith-версия включает SillyTavern-совместимые endpoints (`/api/v1/generate` и др.) прямо в основной сервер на порту 8080. Proxy на порту 5000 не запускается.

---

## 🔌 SillyTavern подключение

| Режим | API Type | Base URL | API Key |
|-------|----------|----------|---------|
| **Proxy** (рекомендуется) | OpenAI | `http://127.0.0.1:5000/v1` | любой |
| **Monolith** | OpenAI | `http://127.0.0.1:8080/v1` | `rtmdk-local` |
| **Monolith** (Text Completion) | Text Completion | `http://127.0.0.1:8080` | — |

Подробнее: [SILLYTAVERN_CONNECTION_GUIDE.md](SILLYTAVERN_CONNECTION_GUIDE.md)

---

## 📚 Документация

| Что нужно | Документ |
|-----------|----------|
| **Главный индекс** | [docs/README.md](docs/README.md) |
| **API справка** | [docs/01_API_REFERENCE.md](docs/01_API_REFERENCE.md) |
| **Production 100K+** | [docs/02_PRODUCTION_GUIDE.md](docs/02_PRODUCTION_GUIDE.md) |
| **Локальный запуск** | [docs/03_LOCAL_SETUP.md](docs/03_LOCAL_SETUP.md) |
| **Docker + Silly Tavern** | [docs/04_DOCKER_SETUP.md](docs/04_DOCKER_SETUP.md) |
| **Тонкая настройка** | [docs/05_FINE_TUNING.md](docs/05_FINE_TUNING.md) |
| **Научная статья** | [docs/06_SCIENTIFIC_ARTICLE.md](docs/06_SCIENTIFIC_ARTICLE.md) |
| **Архитектура** | [docs/08_ARCHITECTURE.md](docs/08_ARCHITECTURE.md) |

## 📊 Результаты

| Метрика | Значение | vs RAG |
|---------|:---:|---|
| **Recall@1** | **95.2%** | +15-35% |
| **Recall@5** | **98.2%** | +13-28% |
| **Latency P95** | 132ms | В 3-15x быстрее |
| **RAM (1K узлов)** | 16 MB | В 3-12x экономнее |

## 🔧 8 Профилей

```python
from rtmdk.config import RTMDKConfig

config = RTMDKConfig.local()       # Персональный ассистент (~16MB)
config = RTMDKConfig.production()  # Продакшен сервер (~50MB)
config = RTMDKConfig.research()    # Максимальная точность (~200MB)
config = RTMDKConfig.enterprise()  # 100K+ узлов, distributed
config = RTMDKConfig.agent()       # Автономный агент
config = RTMDKConfig.legal()       # Юриспруденция (Z3 prover)
config = RTMDKConfig.medical()     # Медицина (Z3 + trust)
config = RTMDKConfig.streaming()   # High-throughput (~3ms)
```

## 🏗️ Архитектура

```
RTMDK v8.0 (25,000+ строк, 75+ файлов, 105+ API)
├── Core: Резонанс, консолидация, HNSW, BM25 (Phase 1-14)
├── Production: Version Control, Attention Tokens (Phase 15)
├── Safety: Symbolic Overlay, UMP, Safety Certifier (Phase 16)
├── Scale: Role Sharding, Swarm Memory (Phase 17)
├── Engrams: Pattern completion, engram decay (Phase 18)
└── Advanced: Offline Dreaming, Causal Traversal, SSM/Mamba,
    Trust Consensus, Neuro-Symbolic Prover (Phase 19)
```

## 📦 Поддерживаемые API

| Провайдер | Переменная |
|-----------|-----------|
| LM Studio (локально, бесплатно) | `RTMDK_API_PROVIDER=lm_studio` |
| OpenRouter (унифицированный) | `RTMDK_API_PROVIDER=openrouter` |
| OpenAI (официальный) | `RTMDK_API_PROVIDER=openai` |
| Anthropic (официальный) | `RTMDK_API_PROVIDER=anthropic` |
| Custom (Groq, Together, LocalAI) | `RTMDK_API_PROVIDER=custom` |

---

*RTMDK v8.0 — Превосходит GraphRAG, Self-RAG и Advanced RAG по точности*
