# RTMDK Documentation Index

> For developers and users navigating the RTMDK documentation.
> Current version: **v8.3.1**

---

## Quick Start (Read First)

| # | Document | Audience | What you'll learn |
|---|----------|----------|-------------------|
| 1 | [README.md](../README.md) | Everyone | Project overview, stats, installation, 5-minute quickstart |
| 2 | [QUICKSTART.md](QUICKSTART.md) | New users | Step-by-step first run, API smoke test |
| 3 | [03_LOCAL_SETUP.md](03_LOCAL_SETUP.md) | Users | Local installation: Python, Docker, SillyTavern |
| 4 | [04_DOCKER_SETUP.md](04_DOCKER_SETUP.md) | DevOps | Docker Compose, .env configuration, production deploy |

---

## API & Integration

| Document | Audience | Content |
|----------|----------|---------|
| [01_API_REFERENCE.md](01_API_REFERENCE.md) | API consumers | Complete endpoint reference (44 endpoints), request/response schemas |
| [09_LANGCHAIN_INTEGRATION.md](09_LANGCHAIN_INTEGRATION.md) | LangChain/LlamaIndex users | Adapters, LCEL pipelines, retrievers, chat history |
| [21_PIPELINE_MIGRATION.md](21_PIPELINE_MIGRATION.md) | Existing users | Migrating from legacy `retrieve_nodes()` to pipeline API |
| [ADR_001_PIPELINE_V83.md](ADR_001_PIPELINE_V83.md) | Architects | ADR: why pipeline architecture replaced monolithic retrieval |

---

## Architecture & Internals

| Document | Audience | Content |
|----------|----------|---------|
| [08_ARCHITECTURE.md](08_ARCHITECTURE.md) | Contributors | Full architecture: 23 phases, all managers, decoupling history |
| [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) | Contributors | Deep dive into 6-stage pipeline: stages, plugins, registry, metrics |
| [20_DOMAIN_MEMORY.md](20_DOMAIN_MEMORY.md) | Contributors | Domain hierarchy, concept lifecycle, bi-temporal facts |
| [SOT_V2_GUIDE.md](SOT_V2_GUIDE.md) | Contributors | Self-Organizing Tokenizer v2: usage, hyperparameters, training |
| [SOT_V2_THEORY.md](SOT_V2_THEORY.md) | Researchers | Mathematical theory behind SOT v2 |
| [MATH_BACKLOG.md](MATH_BACKLOG.md) | Researchers | Mathematical enhancements: Riemannian geometry, conformal prediction, Kalman, spectral clustering |

---

## Operations & Production

| Document | Audience | Content |
|----------|----------|---------|
| [05_FINE_TUNING.md](05_FINE_TUNING.md) | Operators | Config presets, env vars, runtime tuning, 8 profiles |
| [02_PRODUCTION_GUIDE.md](02_PRODUCTION_GUIDE.md) | Architects | Scaling roadmap: 100K→10M nodes, distributed architecture *(planned features)* |
| [25_MONITORING_DEPLOYMENT.md](25_MONITORING_DEPLOYMENT.md) | DevOps | Prometheus + Grafana + Alertmanager stack |
| [DEPLOYMENT.md](DEPLOYMENT.md) | DevOps | Home vs Production deployment comparison |
| [SECURITY.md](../SECURITY.md) | Security reviewers | Security checklist, hardening guide |

---

## Benchmarks & Comparisons

| Document | Audience | Content |
|----------|----------|---------|
| [24_RAG_COMPARISON.md](24_RAG_COMPARISON.md) | Evaluators | RTMDK Pipeline vs traditional RAG (LangChain, LlamaIndex, vector DB) |
| [26_RTMDK_INDUSTRY_COMPARISON.md](26_RTMDK_INDUSTRY_COMPARISON.md) | Evaluators | RTMDK vs industry: Pinecone, Milvus, Weaviate, FAISS |
| [FEATURE_MATRIX.md](FEATURE_MATRIX.md) | Contributors | Experimental feature matrix: integration status, test coverage |
| [scripts/bench_rtmdk_vs_baselines_results.md](../scripts/bench_rtmdk_vs_baselines_results.md) | Evaluators | Internal benchmark results |

---

## Business & Research

| Document | Audience | Content |
|----------|----------|---------|
| [06_SCIENTIFIC_ARTICLE.md](06_SCIENTIFIC_ARTICLE.md) | Researchers / Patent | Full scientific article (patent-pending technology) |
| [ROADMAP.md](ROADMAP.md) | Product / Business | Commercial roadmap: branding, community, monetization |
| [BACKLOG.md](../BACKLOG.md) | Product / Engineering | Development backlog: Tracks 1-5 status, acceptance criteria |

---

## Meta

| Document | Purpose |
|----------|---------|
| [CHANGELOG.md](../CHANGELOG.md) | Release history |
| [MASTER_INDEX.md](MASTER_INDEX.md) | Legacy full index (superseded by this README) |
| [AGENTS.md](../AGENTS.md) | Context for AI coding agents |
| [Values.md](../Values.md) | Parameter calibration reference |

---

## ⚠️ Known Documentation Issues

- `02_PRODUCTION_GUIDE.md` describes **planned** architecture (PQ-64, Raft, distributed sharding) — not yet implemented. See disclaimer at top of file.
- `ROADMAP.md` contains commercial projections that may be outdated. Verify dates.
- `ST_PROXY_SETUP.md` and `../SILLYTAVERN_CONNECTION_GUIDE.md` overlap — the root guide is more recent.

---

## Deleted / Archived Documents

The following documents were removed during the v8.3.1 documentation audit:

| Document | Reason | Action |
|----------|--------|--------|
| `PROGRESS.md` | Historical log (v8.1-v8.2), 873 tests, 52 commits — all superseded | **Deleted** |
| `REFACTORING_PLAN.md` | Plan for v8.1 refactor — completed | **Deleted** |
| `MIGRATION_GUIDE.md` | v8.2.0 → v8.2.1 migration — versions obsolete | **Deleted** |
| `RELEASE_CHECKLIST_v8.3.0.md` | Completed release checklist | **Deleted** |
| `DIALOGUE_SUMMARY_SOT.md` | Outdated SOT dialogue (v8.1) | **Deleted** |
| `kimi-export-*.md` (×2) | AI dialogue logs, 1.4MB — not documentation | **Deleted** |
| `docs/FULL_AUDIT.md` | Core audit for v8.1 — superseded by v8.3.1 fixes | **Deleted** |
| `docs/CODE_REVIEW.md` | Code review for v8.1 — 241 bugs fixed, superseded | **Deleted** |
| `docs/EVALUATION_REPORT_W[1-3].md` | Weekly evaluation reports (738 tests, v8.2 era) | **Deleted** |
| `docs/07_DIALOGUE_EXPORT.md` | Dialogue export for v8.1 | **Deleted** |
| `docs/22_PIPELINE_ARCHITECTURE_DIAGRAM.md` | Mermaid-only diagram — content merged into `PIPELINE_ARCHITECTURE.md` | **Deleted** |

If you need any of these files, they remain in git history prior to the `v8.3.1` tag.
