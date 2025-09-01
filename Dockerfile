# Railway-optimized single-stage build for faster deployment
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

# Copy Railway requirements first for better caching
COPY requirements-railway.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make startup scripts executable
RUN chmod +x /app/scripts/start.py /app/scripts/start_railway.py

# Set environment variables for Railway
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/cache

# Expose port (Railway will set the PORT environment variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health/railway || exit 1

# Run the application using our Railway-optimized startup script
CMD ["python", "/app/scripts/start_railway.py"]