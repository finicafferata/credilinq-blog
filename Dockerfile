# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=2.0.0

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy Railway-optimized requirements first for better caching
COPY requirements-railway.txt ./requirements.txt

# Install Python dependencies with no warnings
RUN pip install --no-cache-dir --user --no-warn-script-location -r requirements.txt

# Production stage
FROM python:3.11-slim

# Labels for metadata
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="CrediLinq AI Content Platform" \
      org.label-schema.description="AI-powered content management and marketing automation platform" \
      org.label-schema.url="https://credilinq.ai" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/credilinq/content-platform" \
      org.label-schema.vendor="CrediLinq" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r credilinq && useradd -r -g credilinq credilinq

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder --chown=credilinq:credilinq /root/.local /home/credilinq/.local

# Copy application code
COPY --chown=credilinq:credilinq . .

# Make startup scripts executable
RUN chmod +x /app/scripts/start.py /app/scripts/start_railway.py

# Copy Prisma schema and generate client
COPY --chown=credilinq:credilinq prisma ./prisma
RUN /home/credilinq/.local/bin/prisma generate || true

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/home/credilinq/.local/bin:$PATH \
    WORKERS=4

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/cache && \
    chown -R credilinq:credilinq /app/logs /app/uploads /app/cache

# Switch to non-root user
USER credilinq

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health/live || exit 1

# Expose port (Railway will override this with the actual PORT)
EXPOSE 8000

# Run the application using our Railway-optimized startup script
CMD ["python", "/app/scripts/start_railway.py"]