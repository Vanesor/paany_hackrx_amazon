# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache and data directories
RUN mkdir -p /app/cache /app/downloads /app/nltk_data && \
    chown -R 1000:1000 /app/cache /app/downloads /app/nltk_data

# Create non-root user for security
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data')"

# Set environment variables
ENV NLTK_DATA=/app/nltk_data
ENV PYTHONPATH=/app
ENV REDIS_URL=redis://redis:6379
ENV CACHE_DIR=/app/cache

# Expose port
EXPOSE 8000

# Health check - updated to use new API endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "final_2:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
