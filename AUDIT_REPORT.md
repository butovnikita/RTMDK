# RTMDK — Отчёт об аудите проекта

> **Обновление 2026-08-01 (2):** закрыты все хвосты, включая гигиенические: `rtmdk_cold_storage/` удалён (расследование: файлы — небитые cold-дампы убитых тестов стабильности; жизненный цикл TieredNodeStore корректен, `__delitem__` чистит файлы; текущая память — 413 semantic-нод, ссылок на дампы нет), битый `myenv/` удалён. Регресс: **1126 passed / 2 skipped / 0 failed**, flake8+black blocking в CI, perf-baseline закоммичен.
>
> **Обновление 2026-08-01:** все проблемы из раздела 4, кроме гигиенических (пп. 8–12), устранены. Подробности — в `CHANGELOG.md` [Unreleased]. Регресс: **1121 passed / 2 skipped / 0 failed**, Playwright E2E всех 11 страниц админки — без замечаний.

**Дата:** 2026-08-01
**Метод:** статический аудит кодовой базы и документации + прогон тестов + реальный запуск сервера с проверкой API.
**Артефакты прогона:** `pytest_run.log`, `server_run.log` (в корне).

---

## 1. Резюме

Проект в целом зрелый и рабочий: пакет `rtmdk/` полностью соответствует AGENTS.md, все backlog-модули v8.3.0 на месте, сервер стартует, память загружается, round-trip (add → query → delete) работает. Тестовый сьют: **1120 passed / 1 failed / 2 skipped** за 115 с. Главные проблемы: рассинхрон версий (код 8.3.0 vs релиз 8.3.1), игнорирование `.env` сервером, неполные зависимости в `pyproject.toml`, мусор и легаси-дубли в корне репозитория.

## 2. Прогон тестов

```
python -m pytest tests/ -q --tb=short -p no:cacheprovider
→ 1 failed, 1120 passed, 2 skipped in 115.37s
```

- Заявлено в README/AGENTS.md: 1112/1118 — фактически **1123** (1120+1+2). Документация отстаёт.
- Упавший тест: `tests/test_admin_panel.py::TestAdminPanelStructure::test_api_base_configurable` — проверяет наличие `API_BASE`/`apiBase` в `admin/src/App.jsx`, но в незакоммиченной версии админки конфигурация вынесена в `server-context.jsx`. **Тест устарел относительно рабочего дерева** (47 dirty/untracked файлов — незавершённая работа над админкой/сервером).
- `tests/conftest.py` фактически пустой (1 строка-комментарий) при 159 тестовых файлах.

## 3. Запуск сервера и содержимое памяти

**Запуск:** `python -m rtmdk` → uvicorn, старт ~9 с, баннер «RTMDK Production API v8.3.0».

**Что лежит в памяти (`~/.rtmdk/memory.json`):**
- Формат: **msgpack + zlib** (несмотря на расширение `.json`), schema 1.0, с `_checksum` — при старте контрольная сумма успешно верифицирована.
- **413 нод**, все в тире `semantic`. Статистика поля: 413 добавлений, 0 запросов (до аудита).
- Содержимое: фрагменты диалогов User/Agent (RU) из сессий разработки + описание самого RTMDK. Ноды богатые: 50 полей (latent_pos, phase, salience, causal_parents, tier, domain, evidence_spans и т.д.).
- SOT: vocab 852/4096, токенизация word. Конфиг: embedding_dim=768, latent_dim=256.
- В `~/.rtmdk/` также: 8 файлов `memory.json.corrupted.*` (история повреждений апр–май 2026), бэкапы, `analytics.db`, `sot_checkpoint.json` (7.3 MB), `memory.msgpack`.
- `rtmdk_cold_storage/` в репо: **1550 msgpack-файлов** (07–09.05.2026) — дампы cold-тиров от тестов стабильности, не используются при текущем запуске (tier_distribution = только semantic).

**Проверенные эндпоинты (все OK):**
| Проверка | Результат |
|---|---|
| `GET /health` | 200: healthy, 413 нод, integrity OK, version 8.3.0 |
| `GET /health/deep` | 200: memory_field/hnsw/embedding_dims(768)/wal/async_index — все ok |
| `POST /v1/memory/query` | 200: релевантные результаты, score ~0.33 |
| `POST /v1/memory/nodes` (add) | 200: нода создана и находится запросом (score 0.37) |
| `DELETE /v1/memory/nodes/{id}` | 200: пробная нода удалена, память не изменилась (413) |
| `GET /v1/memory/nodes` | 200: пагинация, total=413 |
| `GET /v1/sot/status` | 200: enabled, vocab 852/4096 |
| `GET /metrics` | 200: Prometheus-метрики экспортируются |
| Auth | 401 без ключа / с чужим ключом — middleware работает |

LM Studio (`:12345`) на момент проверки был поднят — эмбеддинги и LLM-провайдер доступны (`ai_provider: true`).

## 4. Найденные проблемы

### 🔴 Критичные
1. **Рассинхрон версий.** `pyproject.toml` и `dist/` = **8.3.1**, но `rtmdk/__init__.py` = **8.3.0**, баннер и `/health` сервера = 8.3.0 (4 места в `server/app.py`), AGENTS.md/BACKLOG.md = 8.3.0. CHANGELOG 8.3.1 утверждает, что `__version__` исправлен на 8.3.1 — фикс **не закоммичен** (`git log -S '8.3.1' -- rtmdk/__init__.py` пуст).
2. **Сервер игнорирует `.env`.** Нет `load_dotenv()` ни в `rtmdk/server/app.py`, ни в `start_production.py`. Последствия: `RTMDK_PORT=8081` проигнорирован (сервер поднялся на дефолтном **8080**), `RTMDK_API_KEY` из `.env` не действует — работает дефолтный ключ **`rtmdk-local`**, что при включённой по умолчанию auth (`ENABLE_API_AUTH=true`) — известный статический секрет.

### 🟡 Средние
3. **Неполные зависимости `pyproject.toml`:** core-deps не включают `uvicorn`, `msgpack`, `httpx`, `tenacity`, `strawberry-graphql`, `prometheus-client`, `grpcio` — они импортируются сервером, но живут только в `requirements*.txt`. Установка пакета из wheel не даст работающий сервер.
4. **Легаси-дубли в корне:** ~10 скриптов (`rtmdk_server.py` — 1195 строк, свой сервер на :80801; `rtmdk_st_proxy.py`, `rtmdk_dashboard_ui.py`, `rtmdk_server_ux.py` и др.) дублируют функционал пакета. Разделение осознанное (SillyTavern-режим), но код развивается параллельно.
5. **Устаревший тест админки** (см. §2) + незакоммиченные изменения в `rtmdk/server/app.py`, `graphql_schema.py`, `serialization.py`, тестах.
6. **Расхождения в документации:** `docs/01_API_REFERENCE.md` — 49 endpoints vs `docs/README.md` — 44; тесты 1112/1118/1123; `admin/README.md` не описывает `server.cjs` (порт 3000) и e2e-скрипты; `docs/api/` и `docs/examples/` — пустые каталоги.
7. **Имя `memory.json` вводит в заблуждение** — это zlib+msgpack. Расширение/документация не отражают формат.

### 🟢 Низкие / гигиена
8. Мусор в корне: `err.log`, `server_*.log`, `test_*.log`, `coverage.json`, `fasttext_bootstrap.model`(+`.npy`), `test_entry.py`, `benchmark.py`, дубль `prometheus.yml` (есть в `monitoring/`), файл с кириллицей/битой кодировкой в имени.
9. `admin/` замусорен debug-артефактами: `debug-query.mjs`, `e2e-*.mjs`, 7 PNG-скриншотов (untracked).
10. `rtmdk_github/RTMDK/` — клон репозитория внутри рабочего дерева (исключён из wheel, но лежит в git).
11. `myenv/` — сломанное venv-окружение (нет `python.exe`), используется системный Python 3.10.11.
12. 8 исторических `memory.json.corrupted.*` в `~/.rtmdk/` — повреждения были неоднократно; текущий файл с checksum цел, но стоит понять причину старых повреждений.

## 5. Рекомендации (без выполнения)

1. Поднять `__version__` и баннеры до 8.3.1, закоммитить; синхронизировать AGENTS.md/README (тесты: 1123).
2. Добавить `load_dotenv()` в `app.py`/`start_production.py` **или** явно задокументировать, что `.env` не читается; сменить дефолтный `API_KEY` на обязательный явный.
3. Дополнить `pyproject.toml` extra `server = [...]` реальными серверными зависимостями.
4. Починить/удалить `test_api_base_configurable`; закоммитить или откатить незавершённые изменения админки.
5. Вынести легаси-скрипты корня в `legacy/` или удалить; почистить логи/артефакты, пополнить `.gitignore`.
6. Заполнить `docs/api/` и `docs/examples/` или удалить; обновить `admin/README.md`.
7. Расследовать природу исторических `memory.json.corrupted.*` (auto-save без атомарной записи?).

## 6. Соответствие AGENTS.md

Все заявленные модули существуют и на месте: `engram_cache`, `observability`, `distributed_lock`, `rag_quality`, `explainability`, `safety`, `timeline`, `json_logger`, `async_embedder`, `circuit_breaker` ✅. Структура каталогов соответствует ✅. Расходятся только: версия (8.3.0 → 8.3.1) и число тестов (1112 → 1123).
