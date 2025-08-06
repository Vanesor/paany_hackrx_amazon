#!/bin/bash

# Accuracy RAG System Startup Script
# Optimized for 1.3GB RAM with 30-second timeout

echo "=================================="
echo "  Accuracy RAG System Startup"
echo "=================================="

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    PYTHON_CMD="python"
else
    echo "Running on host system"
    PYTHON_CMD="python3"
fi

# Set environment variables for optimal performance
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Check memory
echo "Checking available memory..."
if command -v free >/dev/null 2>&1; then
    free -h
elif command -v vm_stat >/dev/null 2>&1; then
    vm_stat
fi

echo ""
echo "System Configuration:"
echo "- Max Memory: 1.2GB"
echo "- Timeout: 30 seconds"
echo "- Partial Timeout: 28 seconds"
echo "- Device: CPU (forced for stability)"
echo "- Model: BAAI/bge-base-en-v1.5"
echo "- LLM: gemini-2.0-flash-exp"
echo "- Concurrency: Single request only"
echo ""

# Check if Google API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  Warning: GOOGLE_API_KEY not set"
    echo "   Set it with: export GOOGLE_API_KEY='your-key-here'"
fi

# Check if API token is set
if [ -z "$API_TOKEN" ]; then
    echo "⚠️  Warning: API_TOKEN not set"
    echo "   Set it with: export API_TOKEN='your-secure-token-here'"
fi

echo ""
echo "Starting Accuracy RAG System..."
echo "Access the API at: http://localhost:8000"
echo "Health check: http://localhost:8000/health"
echo ""

# Start the system
exec $PYTHON_CMD main.py
