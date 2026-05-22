# Monitoring Stack

## Components

- **Prometheus** — metrics collection (port 9090)
- **Grafana** — dashboards (port 3000, admin/admin)
- **Alertmanager** — alert routing (port 9093)

## Dashboards

| Dashboard | File | Metrics |
|-----------|------|---------|
| Pipeline | `grafana-dashboard-pipeline.json` | Per-stage latency, circuit breaker states |
| Cost | `grafana-dashboard-cost.json` | Token savings, query distribution |

## Run

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

## Access

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
- Alertmanager: http://localhost:9093
