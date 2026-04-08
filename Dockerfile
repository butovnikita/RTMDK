FROM python:3.10-slim

WORKDIR /app

# System dependencies (curl for healthcheck, gcc for numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY rtmdk_memory_v8.py .
COPY rtmdk_server.py .
COPY rtmdk/ ./rtmdk/

# Create memory directory
RUN mkdir -p /root/.rtmdk

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run server
CMD ["python", "rtmdk_server.py"]
