# Railway-optimized build with LangGraph support
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-railway.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy LangGraph workflows
COPY langgraph_workflows.py ./
COPY langgraph.json ./

# Make startup scripts executable
RUN chmod +x /app/scripts/start.py /app/scripts/start_railway.py

# Set environment variables for Railway + LangGraph
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    LANGGRAPH_API_URL=http://localhost:8001 \
    ENABLE_LANGGRAPH=true

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/cache

# Expose both FastAPI and LangGraph ports
EXPOSE 8000 8001

# Health check for both services
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health/railway || exit 1

# Start both FastAPI and LangGraph services
CMD ["python", "/app/scripts/start_production.py"]
