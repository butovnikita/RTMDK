# RTMDK Production Docker Image
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY docs/ ./docs/

# Create data directories
RUN mkdir -p /app/data/memory \
    /app/data/backups \
    /app/data/embeddings

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    RTMDK_MEMORY_FILE=/app/data/memory/memory.json \
    RTMDK_AUTO_SAVE=60 \
    RTMDK_LM_STUDIO_TIMEOUT=120

# Expose ports
EXPOSE 8080 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "rtmdk_server.py"]
