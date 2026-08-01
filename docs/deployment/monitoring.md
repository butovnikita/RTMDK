# Monitoring

RTMDK ships a complete observability stack: **Prometheus** (metrics, port
9090), **Grafana** (dashboards, port 3000, admin/admin), and **Alertmanager**
(alerts, port 9093). The server exposes Prometheus metrics at `/metrics` and
pipeline-specific metrics at `/v1/memory/pipeline/prometheus`.

## Run the Stack

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Alertmanager: http://localhost:9093

## Dashboards

| Dashboard | File | Metrics |
|-----------|------|---------|
| Pipeline | `monitoring/grafana-dashboard-pipeline.json` | Per-stage latency, circuit breaker states |
| Cost | `monitoring/grafana-dashboard-cost.json` | Token savings, query distribution |

## Metrics Endpoints

```bash
curl http://localhost:8080/metrics                        # Prometheus exposition
curl http://localhost:8080/v1/memory/pipeline/prometheus  # pipeline-specific metrics
```

## Alerting

Alert rules live in `monitoring/prometheus-alerts.yml`; routing config in
`monitoring/alertmanager.yml`. In-app alerting is also available via the
observability module (`sot.observability_enabled`) with webhook, Slack, and
PagerDuty handlers, plus in-memory p50/p95/p99 latency histograms.

## Full Documentation

- [monitoring/README.md — components and dashboards](../../monitoring/README.md)
- [Monitoring Deployment Guide — full setup, alert rules, SLOs](../25_MONITORING_DEPLOYMENT.md)
