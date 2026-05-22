# Docker Deployment

## Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

Configuration via `.env`:
- `RTMDK_HOST=0.0.0.0`
- `RTMDK_PORT=8080`
- `RTMDK_WORKERS=4`

## Home (with SillyTavern)

```bash
docker-compose -f docker-compose.home.yml up -d
```

Services:
- RTMDK server: http://localhost:8080
- SillyTavern proxy: http://localhost:5000

## Health Check

```bash
curl http://localhost:8080/health
curl http://localhost:8080/v1/memory/pipeline/health
```
