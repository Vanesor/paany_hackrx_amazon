# Optimized Dockerfile for 1.3GB RAM Accuracy RAG System
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with optimizations for limited RAM
RUN pip install --no-cache-dir --no-deps numpy==1.23.5 && \
    pip install --no-cache-dir --no-deps torch==2.0.1 && \
    pip install --no-cache-dir --no-deps huggingface-hub==0.16.4 && \
    pip install --no-cache-dir --no-deps transformers==4.30.2 && \
    pip install --no-cache-dir --no-deps sentence-transformers==2.2.2 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create minimal cache directories
RUN mkdir -p /app/model_cache && \
    chown -R 1000:1000 /app

# Create non-root user for security
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Set environment variables for optimal performance
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2

# Set memory limits for Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check for the optimized system
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with optimized settings for 1.3GB RAM
CMD ["python", "-u", "main.py"]
