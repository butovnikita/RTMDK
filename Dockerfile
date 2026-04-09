# RTMDK — CPU-only production image (~200MB)
FROM python:3.10-slim

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY rtmdk_memory_v8.py .
COPY rtmdk_server.py .
COPY rtmdk_server_ux.py .
COPY rtmdk_dashboard_ui.py .
COPY rtmdk_sillytavern_compat.py .
COPY rtmdk/ ./rtmdk/

# Data dirs
RUN mkdir -p /data /root/.rtmdk

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "rtmdk_server.py"]
