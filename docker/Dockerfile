# ─────────────────────────────────────────────────────────────────
# Dockerfile — Travel MLOps Flask API
# Build:  docker build -t travel-mlops-api .
# Run:    docker run -p 5000:5000 travel-mlops-api
# ─────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY flask_api/ ./flask_api/
COPY models/ ./models/

# Expose port
EXPOSE 5000

# Environment variables
ENV FLASK_APP=flask_api/app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "flask_api/app.py"]
