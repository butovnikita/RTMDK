# RTMDK Monitoring Stack Deployment Guide

> Version: 8.3 | Last updated: 2026-05-09

## Overview

RTMDK Pipeline v8.3 includes a complete production observability stack:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Prometheus │◄────│   RTMDK     │────►│  Grafana    │
│  (metrics)  │     │  (server)   │     │ (dashboards)│
└──────┬──────┘     └─────────────┘     └─────────────┘
       │
       ▼
┌─────────────┐
│ Alertmanager│
│  (alerts)   │
└─────────────┘
```

## Quick Start

### 1. Start with Docker Compose

```bash
# Production stack with Prometheus + Grafana + Redis
docker-compose -f docker-compose.prod.yml up -d

# Verify services
docker-compose -f docker-compose.prod.yml ps
```

### 2. Import Grafana Dashboards

```bash
# Pipeline dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana-dashboard-pipeline.json

# Cost analysis dashboard
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana-dashboard-cost.json
```

Or import manually via Grafana UI:
1. Open http://localhost:3000
2. Login: admin / admin
3. Dashboards → Import → Upload JSON file
4. Select `monitoring/grafana-dashboard-pipeline.json`

### 3. Configure Prometheus

Prometheus automatically scrapes RTMDK at `http://rtmdk:8080/v1/memory/pipeline/prometheus`.

Scrape config (already in `docker-compose.prod.yml`):
```yaml
scrape_configs:
  - job_name: 'rtmdk'
    static_configs:
      - targets: ['rtmdk:8080']
    metrics_path: '/v1/memory/pipeline/prometheus'
    scrape_interval: 15s
```

### 4. Configure Alertmanager

Alert rules are in `monitoring/prometheus-alerts.yml`.

Key alerts:
- **PipelineHighLatency**: p95 > 1000ms for 5m → warning
- **PipelineCircuitBreakerOpen**: breaker open for 1m → critical
- **PipelineHighErrorRate**: > 0.1 errors/sec for 5m → warning

Test alerts:
```bash
curl http://localhost:9093/-/healthy
curl http://localhost:9090/api/v1/alerts
```

## Dashboards

### Pipeline Dashboard (`grafana-dashboard-pipeline.json`)

Panels:
- **Queries/sec** — RPS статистика
- **Stage Latency p95** — per-stage latency
- **Circuit Breaker States** — состояние breaker'ов
- **Errors/sec** — ошибки по стадиям
- **Pipeline Health** — таблица состояний
- **Open Breakers** — счётчик открытых breaker'ов
- **Latency Heatmap** — распределение latency

### Cost Dashboard (`grafana-dashboard-cost.json`)

Panels:
- **Cost Per Query (avg)** — средняя стоимость
- **Cost by Stage** — распределение по стадиям
- **Cost vs Latency Scatter** — корреляция
- **Queries by Cost Bucket** — гистограмма
- **Top 10 Most Expensive Queries** — таблица
- **Stage Cost Breakdown** — временной ряд

## Available Metrics

### Counter metrics
- `rtmdk_pipeline_queries_total` — total queries
- `rtmdk_pipeline_errors_total` — errors by stage
- `rtmdk_pipeline_breaker_opens_total` — breaker open events

### Gauge metrics
- `rtmdk_pipeline_breaker_state` — breaker state (0=closed, 1=open, 2=half_open)
- `rtmdk_pipeline_nodes_total` — total nodes in memory

### Histogram metrics
- `rtmdk_pipeline_latency_seconds` — query latency
- `rtmdk_pipeline_stage_latency_seconds` — per-stage latency
- `rtmdk_pipeline_cost_total` — per-query cost

## Health Endpoints

```bash
# Pipeline health
curl http://localhost:8080/v1/memory/pipeline/health

# Pipeline metrics (JSON)
curl http://localhost:8080/v1/memory/pipeline/metrics

# Prometheus metrics
curl http://localhost:8080/v1/memory/pipeline/prometheus

# DAG visualization
curl http://localhost:8080/v1/memory/pipeline/dag

# Plan preview
curl 'http://localhost:8080/v1/memory/pipeline/plan?query=hello&route=fast'
```

## Troubleshooting

### No metrics in Grafana
1. Check Prometheus targets: http://localhost:9090/targets
2. Verify RTMDK server is running: `curl http://localhost:8080/health`
3. Check Prometheus scrape config matches RTMDK address

### High latency alerts
1. Check stage breakdown: `/v1/memory/pipeline/metrics`
2. Identify slowest stage
3. Enable query planner: `pipeline_planner_enabled=True`
4. Enable tiered storage: `tiered_storage_v2_enabled=True`

### Circuit breaker keeps opening
1. Check error rate per stage
2. Review breaker thresholds in config
3. Check logs for stage exceptions
4. Consider increasing `pipeline_breaker_failure_threshold`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RTMDK_PIPELINE_ENABLED` | `false` | Enable pipeline |
| `RTMDK_PIPELINE_PLANNER_ENABLED` | `false` | Enable query planner |
| `RTMDK_PIPELINE_COST_TRACKING` | `false` | Enable cost tracking |
| `RTMDK_TIERED_STORAGE_V2` | `false` | Enable tiered storage v2 |
| `RTMDK_BREAKER_ENABLED` | `true` | Enable circuit breakers |
| `PROMETHEUS_PORT` | `9090` | Prometheus port |
| `GRAFANA_PORT` | `3000` | Grafana port |

## Makefile Targets

```bash
make docker-up      # Start production stack
make docker-down    # Stop production stack
make pipeline-health    # Check pipeline health
make pipeline-metrics   # View pipeline metrics
make pipeline-prometheus # View Prometheus metrics
make diagnose       # Run diagnostics
```
