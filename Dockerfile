# RTMDK Production Docker Image
# Clean production build — No SillyTavern modules.
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-prod.txt .
COPY requirements-sot.txt .
# CPU-only torch first: keeps the production image ~2.3GB smaller and avoids
# runner disk pressure (CUDA wheels are only needed in Dockerfile.gpu)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-prod.txt -r requirements-sot.txt

# Copy the rtmdk package
COPY rtmdk/ ./rtmdk/

# Copy production entry point and shared modules
COPY start_production.py .
COPY legacy/rtmdk_server_ux.py ./legacy/
COPY legacy/rtmdk_dashboard_ui.py ./legacy/
COPY legacy/embedder_lmstudio.py ./legacy/

# Create data directories
RUN mkdir -p /app/data/memory \
    /app/data/backups \
    /app/data/embeddings \
    /app/.rtmdk

# Create non-root user for security
RUN groupadd -r rtmdk && useradd -r -g rtmdk -d /app -s /bin/bash rtmdk \
    && chown -R rtmdk:rtmdk /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    RTMDK_MEMORY_FILE=/app/data/memory/memory.json \
    RTMDK_AUTO_SAVE_INTERVAL=60 \
    RTMDK_LM_STUDIO_TIMEOUT=120 \
    RTMDK_ENABLE_LM_STUDIO=true

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER rtmdk

# Default command — production server
CMD ["python", "start_production.py"]
