# Legacy — SillyTavern Development Servers

> **R8.1 (2026-08-24, audit/risks-2026-08-24): Frozen 2026-08-01 — drift check in `tests/test_repo_health.py:TestLegacyDrift`**
> Этот каталог содержит **замороженные** легаси-скрипты для разработки с интеграцией SillyTavern. Они вынесены из корня репозитория при чистке (2026-08-01). Активная разработка ведётся в пакете `rtmdk/`; production-сервер — `rtmdk/server/app.py` (`python -m rtmdk`).
> **.env:** `start_production.py:9` делает `load_dotenv()` (RTMDK_PORT/API_KEY); `legacy/rtmdk_server.py` — frozen dev-сервер (порт 80801) без `load_dotenv` — используйте лаунчер/прокcи. Новые ST-фичи — только в `rtmdk/` (см. `docs/RISKS.md R8`).

## Содержимое

| Файл | Назначение |
|---|---|
| `rtmdk_server.py` | Монолитный dev-сервер с SillyTavern-эндпоинтами (порт 80801) |
| `rtmdk_server_ux.py` | UX-роутер (импортируется также production-сервером для `/dashboard`) |
| `rtmdk_dashboard_ui.py` | Dashboard-роутер (импортируется production-сервером) |
| `rtmdk_st_proxy.py` | Standalone-прокси между SillyTavern и RTMDK (порт 5000) |
| `rtmdk_sillytavern_compat.py` | Роутер совместимости с API SillyTavern |
| `rtmdk_sillytavern_launcher.py` | Лаунчер «сервер + прокси» одной командой |
| `rtmdk_setup_wizard.py` | Мастер первоначальной настройки |
| `embedder_lmstudio.py` | Эмбеддер через LM Studio (используется тестами) |
| `benchmark.py` | Старый бенчмарк (актуальные — в `scripts/bench_*.py`) |

## Запуск

Из **корня репозитория**:

```bat
start_sillytavern.bat   REM сервер + прокси
start_proxy.bat         REM только прокси
```

или напрямую: `python legacy\rtmdk_server.py`

Пакет `rtmdk` должен быть установлен (`pip install -e .`). Внутренние импорты между модулями каталога работают как прежде (все файлы в одном каталоге). Тесты получают доступ к модулям через `tests/conftest.py` (sys.path shim), production-сервер — через `_ensure_legacy_path()` в `rtmdk/server/app.py`.

## Статус

**Frozen 2026-08-01 (R8.1).** Исправления — только критические. Новые функции SillyTavern-интеграции должны реализовываться в пакете `rtmdk/` (`rtmdk/server/app.py` 48 роутов vs `legacy/rtmdk_server.py` 27 роутов — см. `tests/test_repo_health.py:TestLegacyDrift`). Дрейф ловится CI (`test_legacy_drift.py` logic in `test_repo_health`); при расхождении `legacy` считается отдельным репо-форком.
