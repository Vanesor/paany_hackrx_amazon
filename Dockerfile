# Use Python 3.10 slim image for accuracy-optimized RAG system
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies optimized for BGE-base model
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies with proper order for Python 3.10.12
RUN pip install --no-cache-dir --upgrade pip && \
    # Remove any existing problematic packages
    pip uninstall torch transformers sentence-transformers numpy scipy -y || true && \
    # Install numpy first (critical for other dependencies)
    pip install --no-cache-dir numpy==1.23.5 && \
    # Install PyTorch 2.0.1 (CPU version) to fix uint64 error
    pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    # Install scipy with compatible version
    pip install --no-cache-dir scipy==1.9.3 && \
    # Install transformers and sentence-transformers
    pip install --no-cache-dir transformers==4.30.2 && \
    pip install --no-cache-dir sentence-transformers==2.2.2 && \
    # Install remaining dependencies
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache and model directories for BGE-base
RUN mkdir -p /app/cache /app/downloads /app/models /app/nltk_data && \
    chown -R 1000:1000 /app/cache /app/downloads /app/models /app/nltk_data

# Create non-root user for security
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Download NLTK data and pre-cache models for accuracy
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data')"

# Set environment variables for accuracy-optimized system
ENV NLTK_DATA=/app/nltk_data
ENV PYTHONPATH=/app
ENV CACHE_DIR=/app/cache
ENV MODEL_CACHE_DIR=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
ENV HF_HOME=/app/models

# Expose port
EXPOSE 8000

# Health check for accuracy-optimized API
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with optimized settings for accuracy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--reload"]
