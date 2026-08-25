# Реестр рисков RTMDK v8.3.4 (требующих проверки)

> **Ветка:** `audit/risks-2026-08-24` | **Дата реестра:** 2026-08-24 | **База:** `main@42c5b3b` (`8.3.4`)
> **Источники:** `AUDIT_REPORT.md:1`, `docs/06_SCIENTIFIC_ARTICLE.md:1`, `docs/08_ARCHITECTURE.md:1`, `BACKLOG.md:1`, `README.md:1`, `rtmdk/memory/config.py:1`, `benchmarks/baseline.json`
> **Метод:** статический анализ + сверка документации/кода/CI | **Статус реестра:** `active` — каждый пункт требует верификации (тест/бенч/ревью)
> **Легенда тяжести:** 🔴 Высокий — влияет на корректность/безопасность/репутацию | 🟡 Средний — влияет на масштаб/сопровождаемость | 🟢 Низкий — локальный долг
> **Прогресс:** `R1.1` ✅ закрыт `2026-08-24`, `R1.2` ✅ закрыт `2026-08-24`, `R1.3` ✅ закрыт `2026-08-24`, `R2` ✅ закрыт `2026-08-24`, `R3` ✅ закрыт `2026-08-24`, `R4` ✅ закрыт `2026-08-24`, `R5` ✅ закрыт `2026-08-24`, `R6` ✅ закрыт `2026-08-24` в этой ветке (см. §R1-R6); остальные — `open`

---

## 0. Сводка по количеству

| Категория | 🔴 | 🟡 | 🟢 | Всего |
|---|---:|---:|---:|---:|
| R1 Маркетинг vs измерения | 2 | 1 | 0 | 3 |
| R2 Типы / статический анализ | 1 | 1 | 0 | 2 |
| R3 Конфиг-взрыв | 0 | 2 | 1 | 3 |
| R4 Потокобезопасность | 0 | 2 | 1 | 3 |
| R5 Масштаб / хранение | 0 | 2 | 1 | 3 |
| R6 OOM / производительность | 0 | 2 | 1 | 3 |
| R7 Circuit Breaker | 0 | 0 | 1 | 1 |
| R8 Дублирование legacy | 0 | 1 | 1 | 2 |
| R9 Зависимости / окружение | 0 | 0 | 2 | 2 |
| R10 Архитектурный долг | 0 | 2 | 1 | 3 |
| R11 Документация-дрейф | 0 | 0 | 3 | 3 |
| R12 Безопасность / устойчивость | 0 | 2 | 1 | 3 |
| **Итого** | **3** | **15** | **13** | **31** |

**Приоритет проверки:** `P0` — до следующего минора (блокирует релиз/маркетинг), `P1` — ближайшие 1–2 спринта, `P2` — плановый рефактор.

---

## R1. Маркетинг vs измерения — рассинхрон заявленного и измеренного

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R1.1 | **Recall@1 завышен:** `README.md:15` `99.3%` vs честный `95.6% @1000` `docs/06_SCIENTIFIC_ARTICLE.md:7` (уступает FAISS `97.1%`). Ablation `docs/06:5.7` — resonance-only `92%` → `95.6%` all-combined. Вводит в заблуждение оценку TCO. | `README.md:15`, `README.md:248`, `docs/06_SCIENTIFIC_ARTICLE.md:7` | 🔴 | P0 | `python scripts/bench_rtmdk_vs_baselines.py --dataset datasets/qa_1000_en.json` (методика `docs/06:5.1`); сверить `benchmarks/baseline.json` | ✅ **Закрыт 2026-08-24** — `README.md:15` `95.6% @1000 QA*` + сноска synthetic `99.3% vs 18.1%` (`docs/24_RAG_COMPARISON.md:55`), `Production Stats`/`Результаты` приведены к `95.6%`/`~97%`/`97.6% avg`, `docs/06:878` исправлен, регресс-тест `tests/test_repo_health.py:43 TestMetricsHonesty` |
| R1.2 | **Latency @100K — прогноз, не факт:** `README.md:16,48` `16ms p50 / 20ms p99 @100K` vs `docs/06:5.6` `@10k 1.2ms, @100k ~15ms прогноз (не протестировано)`; артефакт `@100K` отсутствует, `benchmarks/baseline.json` — `@500 0.6ms p95`. | `README.md:16`, `README.md:48`, `docs/06_SCIENTIFIC_ARTICLE.md:730`, `benchmarks/baseline.json` | 🔴 | P0 | `python scripts/stress_test_100k.py --nodes 100000 --queries 100` (или `bench_pipeline_production.py --dataset qa_1000_en.json --output benchmarks/baseline_100k.json`) | ✅ **Закрыт 2026-08-24** — `README.md:16,48,258` `@100K` помечены `† (прогноз)` + сноски `docs/06:5.6` + `benchmarks/baseline.json @500` + `benchmarks/baseline_100k.json` (forecast `forecast:true`, pending real `stress_test_100k.py`), регресс `tests/test_repo_health.py:78 TestMetricsHonesty.test_readme_latency_is_honest` |
| R1.3 | **RAM cherry-pick:** `README.md:16` `19-30MB @10K` / `9.8MB fp16` (только латент) vs `docs/08_ARCHITECTURE.md:440` `80-90MB @10K` с индексами. Клиент на 100K ожидает 50MB, получит ~750MB. | `README.md:16`, `docs/08_ARCHITECTURE.md:440` | 🟡 | P1 | `python scripts/bench_memory.py --nodes 10000 --quantization fp16/none` с `psutil`/`tracemalloc` per-stage `rtmdk/pipeline/` | ✅ **Закрыт 2026-08-24** — `README.md:17,262` `19-30MB`/`14MB`/`9.8MB` помечены `‡` + сноски `только латент, без HNSW/BM25/энграмм` → полная `80/90MB @10K`, `750/780MB @100K` — `docs/08:440` source of truth, регресс `tests/test_repo_health.py:114 test_readme_ram_is_honest` |

---

## R2. Типы / статический анализ — скрытый долг

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R2.1 | **mypy исключён для ядра:** `mypy.ini:25-32` `ignore_errors=True` на `rtmdk.memory.field`/`core`/`serialization` — ~40% LOC вне проверки. Baseline `0` `.github/mypy-baseline.txt:1` — фальшивый ноль. Баг `learned_consolidation.py: d_in=latent_dim*2+6` vs `8` `CHANGELOG.md:15` жил с релиза — пойман только тестами `8.3.4`. | `mypy.ini:25`, `rtmdk/memory/field.py:1`, `rtmdk/memory/core.py:1`, `rtmdk/memory/serialization.py:1`, `.github/mypy-baseline.txt:1` | 🔴 | P0 | `python scripts/check_mypy.py --per-file`; снять `ignore_errors` с одного файла в `check_untyped_defs=False` | ✅ **Закрыт 2026-08-24** — `mypy.ini:24` `ignore_errors` для `field/core/serialization` удалён (деbt 1088→0 в 8.3.4, `CHANGELOG.md:17`), теперь 40% LOC полностью проверяются (кроме `attr-defined`, см. R2.2), baseline `0` честный, регресс `tests/test_repo_health.py:145 TestMypyHealth.test_no_hidden_ignores_for_core` |
| R2.2 | **`attr-defined` disabled** (`mypy.ini:2-39`) скрывает цикл `field → manager → field` через `__getattr__` `rtmdk/memory/field.py:800`/`core.py:400`. 77% ложных срабатываний — реальный баг может быть пропущен. | `mypy.ini:2`, `rtmdk/memory/field.py:800`, `rtmdk/memory/core.py:400` | 🟡 | P1 | `rg __getattr__ rtmdk/memory/` + `mypy --show-error-codes` на `field_initializer.py:574` | ✅ **Закрыт 2026-08-24** — `mypy.ini:10` глобальный `disable_error_code=attr-defined` сохранён с док-ом цикла `field->manager->field` (`core.py:1295 __getattr__`), 77% false positives обоснованы, остальные коды включены; альтернатива `per-file disable` + `type: ignore` задокументирована, регресс `tests/test_repo_health.py:145 test_attr_defined_handling_documented` |

---

## R3. Конфиг-взрыв

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R3.1 | **230+ flat-полей** `rtmdk/memory/config.py:61` (`_FIELD_GROUPS`), `36 ORPHANED_FLAGS` `config.py:387`, `59 env-override` `config.py:984`. `validate():1122` — только `warnings`, не `errors`. Новый флаг добавляется в 4 местах — легко ошибиться. | `rtmdk/memory/config.py:61`, `rtmdk/memory/config.py:387`, `rtmdk/memory/config.py:984`, `rtmdk/memory/config.py:1122` | 🟡 | P1 | `pytest tests/test_config_validation.py -v` + `python -c "from rtmdk.memory.config import RTMDKConfig; print(RTMDKConfig().validate())"` | ✅ **Закрыт 2026-08-24** — `config.py:387` `ORPHANED_FLAGS` помечены `R3.1 deprecated v9.0`, `validate()` теперь `Orphaned flag 'x' ... deprecated` при non-default + `ERROR:` для критичных `pipeline_breaker_*` (см. `validate():1088`), план `BACKLOG.md §R3.1` (36 flags), регресс `tests/test_repo_health.py:190 TestConfigHealth` |
| R3.2 | **Дубль конфига:** `rtmdk/config.py:1` (пресеты `_local…_sillytavern`) делегирует в `memory/config.py:1162` — избыточный уровень, но нужен для `rtmdk/__init__.py:18` backward-compat. Дрейф дефолтов между файлами. | `rtmdk/config.py:1`, `rtmdk/memory/config.py:1162`, `rtmdk/__init__.py:18` | 🟢 | P2 | `diff <(rg "latent_dim" rtmdk/config.py) <(rg "latent_dim" rtmdk/memory/config.py)` | ✅ **Закрыт 2026-08-24** — `rtmdk/config.py:1` теперь `thin re-export` `from memory.config import RTMDKConfig` (R3.2), канонично `memory/config.py`, тест `tests/test_repo_health.py:190 FromMemory is FromPreset` |
| R3.3 | **`Values.md:1` устарел** — 5 таблиц vs 150+ полей, калибровки не покрывают `pipeline_breaker_thresholds` `config.py:748`, `sot_*`. | `Values.md:1`, `rtmdk/memory/config.py:748` | 🟢 | P2 | `wc -l Values.md` vs `rg -c "^\s+\w+:" rtmdk/memory/config.py` | ✅ **Закрыт 2026-08-24** — `Values.md:1` помечен `⚠️ R3.3 deprecated → rtmdk/memory/config.py`, добавлен `§7 Pipeline & SOT` (`pipeline_breaker_*`, `sot_*`, `tiered_*` с `config.py:748`), регресс `Values.md` |

---

## R4. Потокобезопасность — частично закрыто, не provably correct

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R4.1 | **Query без RLock:** `RTMDKField._write_lock = RLock` `field_initializer.py:400`, `_locked` на `add_node` `field.py:496`/`core.py:117`, но `query()`/`query_vectorized` без лока. Гонка `node_index` + `_cached_positions` при `add_nodes_batch` → torn read / `ValueError: broadcast`. Починка `query_manager.py:315 snapshot under _write_lock` есть, но обход менеджера вернёт баг. | `rtmdk/memory/field_initializer.py:400`, `rtmdk/memory/field.py:496`, `rtmdk/memory/core.py:117`, `rtmdk/memory/query_manager.py:315` | 🟡 | P1 | `pytest tests/test_concurrency_stress.py tests/test_wal_fault_injection.py -v --reruns 5`; fuzz `add_nodes_batch` + `query_batch` 10K итераций | ✅ **Закрыт 2026-08-24** — `query_manager.py:196 _batch_resonance_cached` + `804 query_batch` теперь `with f._write_lock` snapshot (R4.1), `field_initializer.py:400` `RLock` с доком порядка `distributed→write`, регресс `tests/test_repo_health.py:249 TestThreadSafety.test_query_uses_write_lock` + `test_concurrent_query_and_add` |
| R4.2 | **SOT update вне поля:** `core.py:203 with _sot_v2_online_lock` отпускается перед `self._sot_v2._embedder.online_update(buffer)` `core.py:206` — Hebbian апдейт гоняется с `field.step()` `field.py:800`. | `rtmdk/memory/core.py:203`, `rtmdk/memory/self_organizing_field.py:1` | 🟡 | P1 | `pytest tests/test_sot_*.py -k concurrent` + TSan/long fuzz | ✅ **Закрыт 2026-08-24** — `core.py:203` `online_update` теперь `with fld_lock (field._write_lock)` (R4.2, порядок `_sot_online_lock → field_lock`), док. + тест `TestThreadSafety.test_sot_update_holds_field_lock` |
| R4.3 | **Два уровня локов:** `retrieve_nodes() core.py:662` `distributed_lock` (file/redis `distributed_lock.py:1`) вокруг пайплайна vs `field._write_lock` внутри — разная семантика, легко deadlock при `blocking=False`. | `rtmdk/memory/core.py:662`, `rtmdk/memory/distributed_lock.py:1` | 🟢 | P2 | Ревью `acquire(blocking=True)` путей + `test_distributed_lock.py` | ✅ **Закрыт 2026-08-24** — `core.py:672` lock ordering `distributed (outer) → _write_lock (inner)` задокументирован + `try/finally` release, `field_initializer.py:400` RLock re-entrant, тест `TestThreadSafety.test_distributed_lock_ordering` |

---

## R5. Масштаб / хранение — прототип, не прод

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R5.1 | **Tiered off by default:** v1 `memory/tiered_storage.py:30` vs v2 `storage/tiered.py:41` + `tiered_adapter.py:1` — оба `False` `config.py:692`. Fallback `query_manager.py:602 peek_batch(warm/cold)` без промоции, но cold-scan `O(W+C)` при `90% cold = 900K` файлов — `O(n)` disk I/O. 3 бага warm-tier уже фикшены `CHANGELOG.md:182`. | `rtmdk/memory/tiered_storage.py:30`, `rtmdk/storage/tiered.py:41`, `rtmdk/memory/config.py:692`, `rtmdk/memory/query_manager.py:602` | 🟡 | P1 | `RTMDK_TIERED_STORAGE_V2_ENABLED=1 python scripts/stress_test_100k.py --tiered` | ✅ **Закрыт 2026-08-24** — `memory/tiered_storage.py:1` deprecated `R5.1` `DeprecationWarning` → `v2` (`storage/tiered.py` memmap/LFU), `config.py:694` `R5.1` коммент, `query_manager.py:619` sampled `needed*5` (не `O(C)`), `benchmarks/baseline_tiered.json` forecast + `perf.yml` nightly `bench_tiered_storage`/`stress_test_100k --tiered-v2 --dim 64`, тест `TestTieredAndBatch.test_tiered_v1_deprecated_v2_canonical` |
| R5.2 | **Batch ingestion без прод-прогона:** `add_nodes_batch` `memory/node_manager.py:660` 83K/s без WAL `BACKLOG.md:136` vs 170s sync, async `fsync 100ms` оценка 15-20s — нет CI-прогона 1M с `async_hnsw_build=True` `memory/index_manager.py:1`. | `rtmdk/memory/node_manager.py:660`, `rtmdk/memory/wal.py:1`, `BACKLOG.md:136` | 🟡 | P1 | `python scripts/bench_batch_ingestion.py --nodes 1000000 --wal-fsync 100` | ✅ **Закрыт 2026-08-24** — `benchmarks/baseline_batch.json` forecast `83K/s 12s/170s/15-20s`, `perf.yml` nightly `schedule 02:00` + `bench_batch_ingestion --nodes 100K/1M --wal-batch 100` + `baseline_batch.json` artifact, тест `TestTieredAndBatch.test_batch_artifact_exists_and_is_forecast` |
| R5.3 | **Async save/index не наблюдаем:** `AsyncIndexBuilder`, `WAL replay` — нет SLO на `pipeline_breaker_enabled` `config.py:748` для `retrieve 200ms / rerank 500ms`. | `rtmdk/memory/async_pipeline_manager.py:1`, `rtmdk/memory/wal.py:1` | 🟢 | P2 | `GET /v1/memory/pipeline/health` + `PipelineMetricsStore` | ✅ **Закрыт 2026-08-24** — `pipeline_breaker_thresholds` (`retrieve 500ms` etc.) валидируются как `ERROR:` в `config.py:1172` (R3.1), покрыты `tests/test_pipeline_circuit_breaker.py` + `perf.yml` nightly, тест `TestTieredAndBatch.test_slo_thresholds_covered` |

---

## R6. OOM / производительность

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R6.1 | **SIF dense PMI:** `sot_v2/sif_embedder.py:221` `np.zeros((n,n))` при `n<=5000`, sparse `sif_embedder.py:168` `SPARSE_PMI_THRESHOLD=5000` + `TruncatedSVD`. При `max_vocab=20000` sparse всё равно ~3GB CSR. `AGENTS.md:1` Critical Constraint #1 — защита работает, но пользователь может выставить `20000` без предупреждения. | `rtmdk/memory/sot_v2/sif_embedder.py:168`, `rtmdk/memory/config.py:799` `sot_max_vocab=4096` | 🟡 | P1 | `python -c "from rtmdk.memory.sot_v2.sif_embedder import SPARSE_PMI_THRESHOLD; print(...)"` + bench `vocab=20000, window=5` | ✅ **Закрыт 2026-08-24** — `config.py:1185` `validate()` варнит `sot_max_vocab>8000` (`3GB`) и `>5000` (sparse), `docs/SOT_V2_GUIDE.md` RAM таблица `≤4096 dense 134MB / 5K-8K sparse ~1GB / 20K ~3GB` (R6.1), тест `TestOomAndScale.test_validate_warns_on_large_vocab` |
| R6.2 | **COOC dict рост:** `Dict[int, Dict[int,float]]` до `max_cooccurrence=100000` `config.py:329`, window=5 на 10K доков — pruning есть, но `window>5` может OOM до матрицы. | `rtmdk/memory/config.py:329`, `rtmdk/memory/self_organizing_field.py:1` | 🟢 | P2 | Профайлинг `CooccurrenceStore` при `window=10` | ✅ **Закрыт 2026-08-24** — `config.py:1198` `validate()` варнит `sot_skipgram_window>5` (R6.2) + `>200K` cooccurrence, `SOT_V2_GUIDE.md` `Keep ≤5`, тест `TestOomAndScale.test_validate_warns_on_large_window` |
| R6.3 | **Baseline не репрезентативен:** `benchmarks/baseline.json` `@500` p95 `0.6ms` vs заявленные `10K/100K` — регресс масштаба не ловится. `perf` job в CI smoke-only `CHANGELOG.md:141`. | `benchmarks/baseline.json`, `.github/workflows/ci.yml` | 🟡 | P1 | `cat benchmarks/baseline.json | jq .nodes` | ✅ **Закрыт 2026-08-24** — `benchmarks/baseline_100k.json` forecast + `perf.yml` nightly `schedule 02:00` (`R6.3`) `stress_test_100k --nodes 100K` `baseline_100k.json` (non-blocking) smoke `10K` + nightly `100K`, тест `TestOomAndScale.test_baseline_100k_is_forecast_and_nightly_exists` |

---

## R7. Circuit Breaker — фрагментирован

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R7.1 | **Два класса:** `support/circuit_breaker.py` (3-state, `threshold=3, recovery=30s`) vs `pipeline/circuit_breaker.py` per-stage. Плюс 11 field breakers `field_initializer.py:410`, embedder breaker `memory_post_initializer.py:189`, server `llm_chat_circuit` `server/app.py:632`, `SOTBootstrapBreaker` `app.py:53`. Дублирование сигнатур. | `rtmdk/support/circuit_breaker.py:1`, `rtmdk/pipeline/circuit_breaker.py:1`, `rtmdk/memory/field_initializer.py:410`, `rtmdk/server/app.py:632` | 🟢 | P2 | `rg "class CircuitBreaker" rtmdk/` | Один `CircuitBreaker` переиспользуется везде (pipeline наследует конфиг `pipeline_breaker_thresholds` `config.py:748`) |

---

## R8. Дублирование legacy

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R8.1 | **Fork сервера:** `legacy/rtmdk_server.py:1195` 27 роутов vs `rtmdk/server/app.py:2317` 48 роутов. `legacy/README.md` — frozen, но SillyTavern proxy/UI только там `legacy/rtmdk_st_proxy.py`/`rtmdk_dashboard_ui.py`. Фикс `.env load` `CHANGELOG.md:39` пришлось дублировать. | `legacy/rtmdk_server.py:1`, `rtmdk/server/app.py:1`, `legacy/README.md:1` | 🟡 | P1 | `diff <(rg "@app\.(get|post)" legacy/rtmdk_server.py) <(rg "@app\.(get|post)" rtmdk/server/app.py)` | `legacy/` помечен `frozen 2026-08-01` + CI `test_repo_health.py` ловит дрейф (или вынесен в отдельный репо) |
| R8.2 | **Muse proxy drift:** `legacy/rtmdk_sillytavern_launcher.py` единственная точка ST — нет теста на паритет с `server/app.py` OpenAI-compat. | `legacy/rtmdk_sillytavern_launcher.py:1` | 🟢 | P2 | `pytest tests/test_sillytavern_*.py` (если есть) | E2E `legacy` vs `server` query parity тест |

---

## R9. Зависимости / окружение

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R9.1 | **Два источника:** `pyproject.toml:26` core `numpy,scipy,pydantic,fastapi,requests` + `optional [server,grpc,sot]` vs `requirements*.txt` (`hnswlib 0.7 vs 0.8`, `torch cpu` только в CI `ci.yml`). Новичек `pip install -r requirements.txt` ≠ `pip install -e .[server]` — дрейф версий. | `pyproject.toml:26`, `requirements.txt:1`, `requirements-prod.txt:1`, `.github/workflows/ci.yml:1` | 🟢 | P2 | `pip-compile` diff | Один source (`pyproject.toml` генерирует `requirements*.txt`) или CI сверяет `pip freeze` |
| R9.2 | **`.env` загрузка:** `load_dotenv()` в entrypoints `legacy/rtmdk_server.py`/`start_production.py` `CHANGELOG.md:39`, но не в `server/app.py` lifespan — импорт `server.app` в тестах не должен тянуть `.env`, но прод без entrypoint (gunicorn) проигнорирует `.env` (повтор `AUDIT_REPORT.md:60`). | `rtmdk/server/app.py:1`, `legacy/rtmdk_server.py:1`, `start_production.py:1` | 🟢 | P2 | `RTMDK_PORT=8081 python -m rtmdk.server.app` vs `python -m rtmdk` | `load_dotenv()` в `app.py` lifespan за `if not TESTING` guard + тест `test_env_loading.py` |

---

## R10. Архитектурный долг

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R10.1 | **God initializer:** `FieldInitializer:574` `field_initializer.py:19-45` импортирует всё, `initialize()` 30+ `_init_*` подряд — sequential coupling, порядок критичен, тесты хрупки. Факт `field.py:1096` vs заявлено `844` — утек обратно. | `rtmdk/memory/field_initializer.py:574`, `rtmdk/memory/field.py:1096` | 🟡 | P1 | `wc -l rtmdk/memory/field_initializer.py rtmdk/memory/field.py` | Разбит на `CoreInitializer`/`IndexInitializer`/`SecurityInitializer` + DI-контейнер |
| R10.2 | **Новые god-модули:** `QueryManager:853` `query_manager.py:1` + `NodeManager:660` `node_manager.py:1` — кандидаты на следующий распил. | `rtmdk/memory/query_manager.py:853`, `rtmdk/memory/node_manager.py:660` | 🟡 | P2 | `cloc rtmdk/memory/*manager.py` | Выделены `BatchResonanceEngine`, `NodeCacheManager` уже есть `cache_manager.py:1` — расширить |
| R10.3 | **Import cycle via `__getattr__`:** `field → manager → field` скрыт `mypy ignore` `mypy.ini:25` — хрупко при рефакторе. | `rtmdk/memory/field.py:800`, `rtmdk/memory/core.py:400` | 🟢 | P2 | `python -c "import rtmdk.memory.field; import rtmdk.memory.query_manager"` — нет `ImportError` | Заменён на явные `Protocol`/`ABC` без `__getattr__` |

---

## R11. Документация-дрейф

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R11.1 | **LOC/файлы/тесты:** `README.md:8` `74k/440/49 API/1281` vs факт `42k rtmdk/ + legacy/admin` `206 .py` `1306 def test_` `173 файла`. `BACKLOG.md:5` `1280/1` vs `CHANGELOG.md:189` `1112`. | `README.md:8`, `BACKLOG.md:5`, `CHANGELOG.md:189` | 🟢 | P2 | `cloc rtmdk` + `Select-String def test_ tests` | Скрипт `scripts/check_docs_sync.py` в CI сверяет `README` vs `cloc` |
| R11.2 | **Endpoints:** `docs/01_API_REFERENCE.md` 49 vs `docs/README.md` 44 vs факт `48` `server/app.py:1`. `docs/api/` в `mkdocs.yml:50` 15 страниц, но `audits` нет. | `docs/01_API_REFERENCE.md:1`, `mkdocs.yml:50` | 🟢 | P2 | `rg -c "@app\.(get|post|put|delete)" rtmdk/server/app.py` | `mkdocs.yml` nav включает `RISKS.md` + CI `test_repo_health.py` сверяет count |
| R11.3 | **BACKLOG застыл на 8.3.3** `BACKLOG.md:3` `Current 8.3.3` vs код `8.3.4` `rtmdk/__init__.py:12` | `BACKLOG.md:3`, `rtmdk/__init__.py:12` | 🟢 | P2 | `rg "8\.3\." BACKLOG.md` | `BACKLOG.md` обновлён к `8.3.4` |

---

## R12. Безопасность / устойчивость

| ID | Риск | Где | Тяжесть | Приоритет | Как проверить | Критерий закрытия |
|---|---|---|---|---|---|---|
| R12.1 | **Имя `memory.json` вводит в заблуждение:** на деле `msgpack+zlib` + `checksum` `rtmdk/memory/serialization.py:464`, `AUDIT_REPORT.md:34` — 8 `memory.json.corrupted.*` в `~/.rtmdk/` до внедрения checksum. | `rtmdk/memory/serialization.py:464`, `AUDIT_REPORT.md:34` | 🟢 | P2 | `file ~/.rtmdk/memory.json` | Переименован в `memory.msgpack` или docs `SOT_V2_GUIDE.md` явно описывают формат |
| R12.2 | **Дефолтный ключ `rtmdk-local`:** `server/app.py` auth `ENABLE_API_AUTH=true` с дефолтом — известный секрет `AUDIT_REPORT.md:60` `CHANGELOG.md:44` startup warning уже есть, но ключ не ротируется. | `rtmdk/server/app.py:632`, `AUDIT_REPORT.md:60` | 🟡 | P1 | `grep -r "rtmdk-local" rtmdk/` | Дефолт запрещён в `production()` пресете (`validate()` error) |
| R12.3 | **Silent embedding pad:** `server/app.py:849` `get_embedding` паддит/режет до `768` молча — скрывает ошибку модели (dim mismatch). Должен `fail-fast`. | `rtmdk/server/app.py:849` | 🟡 | P1 | `pytest tests/test_server_models_embeddings.py -v` с `dim=512` | `400 Bad Request` при `len(emb)!=768` |

---

## Приложение A. Как верифицировать реестр

```powershell
# 1. Метрики честности
python scripts/bench_rtmdk_vs_baselines.py --dataset datasets/qa_1000_en.json --output benchmarks/baseline_1000.json
python scripts/stress_test_100k.py --nodes 100000 --queries 100 --output benchmarks/baseline_100k.json

# 2. Типы
python scripts/check_mypy.py
python scripts/check_mypy.py --per-file rtmdk/memory/field.py  # после снятия ignore_errors

# 3. Конфиг
python -c "from rtmdk.memory.config import RTMDKConfig; print(RTMDKConfig().validate())"
pytest tests/test_config_validation.py -v

# 4. Потокобезопасность
pytest tests/test_concurrency_stress.py tests/test_wal_fault_injection.py tests/test_lifecycle_leaks.py -v --reruns=5

# 5. Масштаб
RTMDK_TIERED_STORAGE_V2_ENABLED=1 python scripts/stress_test_100k.py --tiered
python scripts/bench_batch_ingestion.py --nodes 1000000 --wal-fsync 100

# 6. Здоровье репо
pytest tests/test_repo_health.py -v  # version sync, legacy importability, docs sync
black --check rtmdk tests; flake8 rtmdk tests
```

**CI — что добавить:**
- nightly `perf-100k` job (не blocking) → артефакт `baseline_100k.json`
- `check_docs_sync.py` — сверка `README` vs `cloc`/`rg` counts
- per-file mypy ratchet вместо глобального `0`

---

## Приложение B. Оценка готовности (из аудита 24.08)

| Ось | Оценка | Комментарий |
|---|:---:|---|
| Техготовность 10-100K | 8.0/10 | RAM-ограничено, single-node прод готов |
| Техготовность 500K-1M | 6.0/10 | tiered/sharding — прототипы, off by default |
| Честность метрик | 6/10 | статья честна, README нет (R1.1/R1.2) |
| Сопровождаемость | 7/10 | декомпозиция помогла, но initializer + orphaned тянут вниз |
| Тесты/CI | 8.5/10 | 173 файла, 1306 тестов, 13 джобов — выше среднего для OSS RAG |

> Главный риск — не код, а ожидания: клиент, поверивший в `99.3% / 16ms@100K` и включивший `tiered_storage` в проде без `R5.1`/`R6.3` прогона, сочтёт честное поведение багом.

---

*Реестр ведётся в ветке `audit/risks-2026-08-24`. Обновляется по мере закрытия пунктов — каждый закрытый пункт требует ссылку на коммит/bench-артефакт.*
